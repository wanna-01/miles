# Ash rollout backend

The Ash rollout backend delegates one complete prompt group to an external
agent rollout service while Miles remains responsible for training. It is
intended for rollout strategies that need to grow branches, checkpoint and
restore environments, or otherwise coordinate a group of related agent
trajectories outside the standard per-sample generation loop.

## Ownership boundary

Miles owns:

- prompt-group and sample-slot allocation;
- the SGLang model and Session Server endpoints exposed to the rollout service;
- the expected policy weight version and sampling configuration;
- validation and conversion of returned trajectories into `Sample` objects;
- reward calculation when the rollout service does not provide a reward; and
- advantage calculation, loss construction, optimization, and weight updates.

The Ash-compatible service owns:

- the agent loop and tool execution;
- environment creation, checkpointing, restoration, and cleanup;
- branch selection and search scheduling within the supplied budgets; and
- assembly of complete token-aligned trajectories for the allocated sample
  slots.

## Request lifecycle

`AshRolloutFn` implements the class-based Miles rollout interface. For each
prompt group it:

1. allocates the sample slots already reserved by Miles;
2. submits an `AshRolloutRequest` with `POST /rollout-groups`;
3. polls `GET /rollout-groups/{rollout_job_id}` until the job is terminal;
4. validates group identity, slot ownership, policy version, messages, and
   token boundaries;
5. imports each returned trajectory into a Miles `Sample`; and
6. sends `DELETE /rollout-groups/{rollout_job_id}` after consuming the result,
   or when local processing fails or times out, so Ash can cancel unfinished
   work and release the complete trajectory record.

The service receives a model endpoint for ordinary generation. When a Miles
Session Server endpoint is supplied, the service can use it for multi-turn
generation with exact token and trajectory tracking.

`ash-rollout-v1` is a strict contract on both sides: unknown fields are
rejected instead of ignored. Any wire-format extension therefore requires a
coordinated protocol update rather than a one-sided optional field.

## Trajectory contract

The wire models live in `protocol.py`. A returned trajectory must include:

- the complete ordered message history, including matched assistant tool calls
  and tool results;
- the exact token sequence and the prompt boundary;
- one `generated_span` for every assistant generation call;
- the input and output token IDs for each generated span;
- the policy weight version used by every generation call; and
- rollout log-probabilities when Miles requests them.

Only assistant-generated spans receive `loss_mask=1`. Prompt tokens, tool
results, and other environment-provided context remain part of the model input
but do not contribute directly to the policy loss.

## Configuration

Select the backend and provide the rollout service URL:

```bash
--rollout-function-path miles.rollout.ash.rollout_fn.AshRolloutFn \
--ash-rollout-base-url http://ash-rollout-service:PORT
```

Miles derives the model and Session Server endpoints from the active rollout
deployment by default. They can be overridden when the Ash service must use
separately advertised addresses:

```bash
--ash-rollout-model-endpoint http://model-router:PORT \
--ash-rollout-session-server-endpoint http://session-server:PORT
```

The `--ash-rollout-max-model-calls`, `--ash-rollout-max-tool-calls`, and
`--ash-rollout-timeout-seconds` options bound the work allowed for one prompt
group. HTTP request timeout and polling cadence are configured separately.

## Current scope

The first protocol version supports training rollout with a fixed number of
returned samples equal to `n_samples_per_prompt`. Evaluation, multimodal
prompts, and early return with fewer samples are not yet supported. Per-task
environment configuration is service-owned in this version rather than copied
from each Miles sample.

Do not enable rollout routing replay, indexer replay, or rollout sampling-mask
features with this backend until their per-token payloads are represented in
the Ash protocol. The exact-token and generated-span requirements are part of
the correctness boundary and must not be replaced by text retokenization.
