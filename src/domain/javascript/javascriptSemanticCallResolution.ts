import * as t from "@babel/types";

import type {
  JavaScriptSemanticArgumentFlow,
  JavaScriptSemanticCallable,
} from "./javascriptSemanticIr.js";
import { compareUnicodeCodePoints } from "../unicodeCodePointOrder.js";
import { semanticCallableIdForNode } from "./javascriptSemanticProjection.js";
import { semanticBindingAtRead } from "./javascriptSemanticMutationInitializers.js";
import {
  unwrapJavaScriptExpression,
  runtimeFunctionParameters,
} from "./javascriptAstValues.js";
import {
  resolveSemanticBindingState,
  semanticStableInitializer,
  type JavaScriptSemanticAnalysisState,
  type JavaScriptSemanticBindingState,
} from "./javascriptSemanticState.js";
import { range } from "./javascriptStaticAnalysisHelpers.js";

/** Resolution of one call target within the currently analyzed source. */
export interface LocalCallableResolution {
  readonly callableIds: readonly string[];
  readonly complete: boolean;
  readonly reason: string;
  /** The resolution crosses a local alias cycle and is reported as uncertain. */
  readonly cycleDependent?: true;
}

/** Shared per-analysis memory for local call-target resolution. */
export interface CallResolutionMemory {
  readonly bindings: Map<
    JavaScriptSemanticBindingState,
    BindingResolutionEntry
  >;
}

type BindingResolutionEntry =
  | { readonly status: "computing" }
  | {
      readonly status: "provisional";
      readonly resolution: LocalCallableResolution;
    }
  | {
      readonly status: "settled";
      readonly resolution: LocalCallableResolution;
    };

interface CallResolutionContext {
  readonly state: JavaScriptSemanticAnalysisState;
  readonly callableById: ReadonlyMap<string, JavaScriptSemanticCallable>;
  readonly memory: CallResolutionMemory;
  readonly bindingCache: WeakMap<t.Identifier, LocalCallableResolution>;
}

/** Inputs needed to resolve one local call target. */
export interface ResolveLocalCallablesInput {
  readonly node: t.Node;
  readonly state: JavaScriptSemanticAnalysisState;
  readonly callableById: ReadonlyMap<string, JavaScriptSemanticCallable>;
  readonly bindingCache: WeakMap<t.Identifier, LocalCallableResolution>;
  /** Omit for a one-off resolution; pass one memory per analysis pass to reuse work. */
  readonly resolutionMemory?: CallResolutionMemory;
}

export const createCallResolutionMemory = (): CallResolutionMemory => ({
  bindings: new Map(),
});

/** Resolve local callable candidates without following dynamic properties. */
export const resolveLocalCallables = (
  input: ResolveLocalCallablesInput,
): LocalCallableResolution => {
  const memory = input.resolutionMemory ?? createCallResolutionMemory();
  const base = {
    state: input.state,
    callableById: input.callableById,
    memory,
  };
  const first = resolveCallables(input.node, {
    ...base,
    bindingCache: input.bindingCache,
  });
  if (!settleMemory(base)) return first;
  // A provisional cycle entry changed, so per-read results derived from the
  // earlier candidate sets are stale. Re-read with the settled fixpoint only.
  return resolveCallables(input.node, {
    ...base,
    bindingCache: new WeakMap(),
  });
};

const resolveCallables = (
  node: t.Node,
  context: CallResolutionContext,
): LocalCallableResolution => {
  // `unwrapJavaScriptExpression` strips all transparent layers iteratively,
  // so a single recursion step suffices (no per-layer stack growth).
  const unwrapped = unwrapJavaScriptExpression(node).node;
  if (unwrapped !== node) return resolveCallables(unwrapped, context);
  const direct = semanticCallableIdForNode(node);
  if (direct !== null && context.callableById.has(direct))
    return {
      callableIds: [direct],
      complete: true,
      reason: "Direct local callable.",
    };
  if (t.isIdentifier(node)) return resolveIdentifier(node, context);
  if (t.isConditionalExpression(node) || t.isLogicalExpression(node))
    return resolveAlternatives(
      t.isConditionalExpression(node) ? node.consequent : node.left,
      t.isConditionalExpression(node) ? node.alternate : node.right,
      context,
    );
  return {
    callableIds: [],
    complete: false,
    reason: `Unsupported ${node.type} call target.`,
  };
};

const resolveIdentifier = (
  node: t.Identifier,
  context: CallResolutionContext,
): LocalCallableResolution => {
  const cached = context.bindingCache.get(node);
  if (cached !== undefined) return cached;
  const binding = resolveSemanticBindingState(context.state, node, node.name);
  if (binding === undefined)
    return {
      callableIds: [],
      complete: false,
      reason: `Unresolved call target ${node.name}.`,
    };
  const read = semanticBindingAtRead(
    binding,
    node,
    context.state.parentsByNode,
  );
  const candidates = bindingResolution(read.binding, context);
  // Knowing every local candidate does not prove that any write replaced the
  // incoming value, or that a write in another activation precedes this read.
  const covered =
    read.hasDefiniteInitializer ||
    semanticStableInitializer(binding, context.state) !== undefined;
  const resolution =
    candidates.complete && !covered
      ? {
          ...candidates,
          complete: false,
          reason: `Call target ${binding.name} may retain an incoming value or an unordered assignment.`,
        }
      : candidates;
  // Key by the immutable read, not binding identity: before/after snapshots
  // differ. The candidate set is the settled per-binding fixpoint, so the
  // read adjustment is deterministic and always cacheable.
  context.bindingCache.set(node, resolution);
  return resolution;
};

const resolveAlternatives = (
  left: t.Node,
  right: t.Node,
  context: CallResolutionContext,
): LocalCallableResolution => {
  const candidates = [left, right].map((candidate) =>
    resolveCallables(candidate, context),
  );
  return {
    callableIds: uniqueCallableIds(
      candidates.flatMap(({ callableIds }) => callableIds),
    ),
    complete: candidates.every(({ complete }) => complete),
    reason: "Conditional or logical call target has multiple candidates.",
    ...(candidates.some((candidate) => candidate.cycleDependent)
      ? { cycleDependent: true as const }
      : {}),
  };
};

const bindingResolution = (
  binding: JavaScriptSemanticBindingState,
  context: CallResolutionContext,
): LocalCallableResolution => {
  const existing = context.memory.bindings.get(binding);
  if (existing !== undefined) {
    // Re-entering a binding that is still being expanded is the alias cycle
    // cut: the contribution arrives through the binding's own expansion.
    if (existing.status === "computing")
      return {
        callableIds: [],
        complete: false,
        cycleDependent: true,
        reason: `Call target initializer cycle at ${binding.name}.`,
      };
    return existing.resolution;
  }
  context.memory.bindings.set(binding, { status: "computing" });
  const resolution = computeBindingResolution(binding, context);
  // Cycle-crossing expansions start provisional: branches cut by re-entry
  // gain their full contribution once the re-entered binding settles.
  context.memory.bindings.set(
    binding,
    resolution.cycleDependent
      ? { status: "provisional", resolution }
      : { status: "settled", resolution },
  );
  return resolution;
};

const computeBindingResolution = (
  binding: JavaScriptSemanticBindingState,
  context: CallResolutionContext,
): LocalCallableResolution => {
  if (binding.initializers.length === 0)
    return missingBindingInitializer(binding);
  const candidates = binding.initializers.map(
    (initializer): LocalCallableResolution =>
      initializer.projection.length === 0
        ? resolveCallables(initializer.node, context)
        : {
            callableIds: [],
            complete: false,
            reason: `Projected call target ${binding.name} is unresolved.`,
          },
  );
  const callableIds = uniqueCallableIds(
    candidates.flatMap((candidate) => candidate.callableIds),
  );
  return {
    callableIds,
    complete: candidates.every(({ complete }) => complete),
    reason:
      callableIds.length === 1 && candidates.every(({ complete }) => complete)
        ? "Unique local callable target."
        : `Call target ${binding.name} has ambiguous or unsupported assignments.`,
    ...(candidates.some((candidate) => candidate.cycleDependent)
      ? { cycleDependent: true as const }
      : {}),
  };
};

/**
 * Drive provisional cycle entries to the least fixpoint. Recomputing a
 * provisional binding consults only current entry values, so candidate sets
 * grow monotonically and a pass without changes settles every entry.
 */
const settleMemory = (base: {
  state: JavaScriptSemanticAnalysisState;
  callableById: ReadonlyMap<string, JavaScriptSemanticCallable>;
  memory: CallResolutionMemory;
}): boolean => {
  let changed = false;
  for (;;) {
    const provisional = [...base.memory.bindings].filter(
      (
        pair,
      ): pair is [
        JavaScriptSemanticBindingState,
        Extract<BindingResolutionEntry, { status: "provisional" }>,
      ] => pair[1].status === "provisional",
    );
    if (provisional.length === 0) return changed;
    let passChanged = false;
    for (const [binding, entry] of provisional) {
      if (base.memory.bindings.get(binding) !== entry) continue;
      // Fresh read cache: cached per-read results derived from earlier
      // candidate sets must not feed the recompute.
      const fresh = computeBindingResolution(binding, {
        ...base,
        bindingCache: new WeakMap(),
      });
      if (!sameResolution(fresh, entry.resolution)) {
        base.memory.bindings.set(binding, {
          status: "provisional",
          resolution: fresh,
        });
        passChanged = true;
      }
    }
    if (!passChanged) {
      for (const [binding, entry] of provisional)
        base.memory.bindings.set(binding, {
          status: "settled",
          resolution: entry.resolution,
        });
      return changed;
    }
    changed = true;
  }
};

const sameResolution = (
  left: LocalCallableResolution,
  right: LocalCallableResolution,
): boolean =>
  left.complete === right.complete &&
  left.cycleDependent === right.cycleDependent &&
  left.reason === right.reason &&
  left.callableIds.length === right.callableIds.length &&
  left.callableIds.every((id, index) => id === right.callableIds[index]);

const missingBindingInitializer = (
  binding: JavaScriptSemanticBindingState,
): LocalCallableResolution => ({
  callableIds: [],
  complete: false,
  reason:
    binding.kind === "import"
      ? `Call target ${binding.name} is imported from outside this source.`
      : `Call target ${binding.name} has no local initializer.`,
});

/** Find all bindings introduced by one positional parameter pattern. */
export const parameterBindings = (
  callable: JavaScriptSemanticCallable,
  index: number,
  state: JavaScriptSemanticAnalysisState,
): JavaScriptSemanticBindingState[] => {
  if (callable.bodyScopeId === null) return [];
  const scope = state.scopesById.get(callable.bodyScopeId);
  if (scope === undefined) return [];
  return [...scope.bindings.values()]
    .filter(
      (binding) =>
        binding.kind === "parameter" &&
        binding.definitions.some(
          ({ kind, location }) =>
            kind === "parameter" &&
            parameterAcceptsArgument(callable, location, index, state),
        ),
    )
    .sort((left, right) =>
      compareUnicodeCodePoints(left.bindingId, right.bindingId),
    );
};

const parameterAcceptsArgument = (
  callable: JavaScriptSemanticCallable,
  location: JavaScriptSemanticArgumentFlow["parameterLocation"],
  argumentIndex: number,
  state: JavaScriptSemanticAnalysisState,
): boolean => {
  const node = state.callableNodesById.get(callable.callableId);
  if (!t.isFunction(node)) return false;
  const parameters = runtimeFunctionParameters(node);
  const parameterIndex = parameters.findIndex((parameter) =>
    rangeContains(range(parameter), location),
  );
  if (parameterIndex === argumentIndex) return true;
  return (
    parameterIndex >= 0 &&
    parameterIndex < argumentIndex &&
    t.isRestElement(parameters[parameterIndex])
  );
};

const rangeContains = (
  outer: JavaScriptSemanticArgumentFlow["parameterLocation"],
  inner: JavaScriptSemanticArgumentFlow["parameterLocation"],
): boolean =>
  (outer.start.line < inner.start.line ||
    (outer.start.line === inner.start.line &&
      outer.start.column <= inner.start.column)) &&
  (outer.end.line > inner.end.line ||
    (outer.end.line === inner.end.line &&
      outer.end.column >= inner.end.column));

const uniqueCallableIds = (values: readonly string[]): string[] =>
  [...new Set(values)].sort(compareUnicodeCodePoints);
