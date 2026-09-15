# Forehand overhead clear — technique rubric

You are given measurements from a **side-on 2D camera**. Elbow angle, contact height/forwardness,
racket shaft angle, and timing are direct 2D image-plane measurements and reliable. Shoulder/hip
**rotation, X-factor, and the hip→shoulder→racket sequence come from monocular 3D** estimation —
`shoulder_rotation_deg_prep_end`/`_contact`, `hip_rotation_deg_prep_end`/`_contact`,
`x_factor_deg_prep_end`/`_contact`, `trunk_lean_3d_deg_contact`, and `sequence_hip_ms`/
`sequence_shoulder_ms`/`sequence_racket_ms` in each swing's `metrics` in `metrics.json` — good enough to
say "the hips turned about 60°" or "hips turned before the shoulders", not precise enough for ±5° claims.
Always quote 3D rotation numbers as approximate, and prefer citing the reference clips' numbers over a
specific claimed target. If a swing has none of these fields at all (older run, or `--skip-body3d`) or
they're all `null` (check `measurement_quality` and each swing's evidence), fall back to the 2D
`shoulder_width_ratio`/`hip_width_ratio` proxy and say "appears" rather than stating a degree value.

## Preparation
- Sideways stance to the net (small `shoulder_width_ratio` relative to a squared-up stance; in 3D,
  `shoulder_rotation_deg_prep_end` roughly 70–90°, `hip_rotation_deg_prep_end` somewhat less — a positive
  `x_factor_deg_prep_end`, i.e. the trunk coiled).
- Racket-arm elbow up and bent around 90°, racket head dropped behind the body ("back-scratch" position
  — this is `prep_end` in the metrics).
- **Non-racket arm raised** toward the shuttle — used for balance, timing, and to help initiate rotation.
- Weight loaded on the back (racket-side) foot.

## Kinetic chain (legs → hips → trunk → shoulder → elbow → forearm pronation → contact)
- The hips should start rotating first, the shoulders follow, the racket last — check
  `sequence_hip_ms`/`sequence_shoulder_ms`/`sequence_racket_ms` (`hip < shoulder < racket`, each roughly
  20–80ms apart is a good sign; simultaneous or reversed suggests an arm-only swing with no rotational
  power). These are timestamps in ms relative to the swing's preparation start, over the
  prep_start→contact window.
- The elbow stays back/bent while the trunk turns; it should not straighten and swing forward until the
  trunk has already turned toward the net. If shoulders are already near-square
  (`shoulder_rotation_deg_prep_end` small) very early relative to contact, the player may be "arming" the
  shot.
- By contact, shoulders should be close to square to the net (`shoulder_rotation_deg_contact` roughly
  ≤ 20°).

## Contact
- **High, above the head, and slightly in front** of the body: `contact_height_vs_nose` should be
  positive (racket above the nose), and `contact_forward` positive and meaningfully forward — coaching
  sources describe roughly half a metre in front of the body, which in these torso-length-normalized
  units is very roughly +0.2 to +0.5, but prefer whatever the reference clips actually show over this
  number.
- Arm **nearly** extended but not locked: `elbow_angle_contact` roughly 150–170°. Too straight (close to
  180°) means the shoulder did all the work with no forearm/wrist contribution; too bent (well under
  140°) limits reach and rotational power transfer.
- Racket shaft close to vertical at contact for a clear (`racket_shaft_angle_contact` near 0).
- For `mode="shadow"` swings (no shuttle visible near the racket), judge the position at peak racket
  speed as the *intended* contact point. Never comment on timing relative to the shuttle or where the
  shuttle went for a shadow swing — there is no shuttle data to support that.

## Follow-through
- Racket continues down and across the body after contact; weight transfers to the front (net-side)
  foot (`weight_transfer` = true, `front_foot` changes between prep and contact).
- For a jump clear: a scissor-kick landing (feet switch in the air) is normal and not itself a fault.
- The swing should not visibly stop dead right at contact — some continued racket travel into
  `follow_end` is expected.

## Common faults to look for
- Contact behind the head or too low (`contact_height_vs_nose` ≤ 0, or `contact_forward` ≤ 0).
- Elbow too bent or fully locked at contact.
- No hip/trunk turn — facing the net throughout the swing (`shoulder_rotation_deg_prep_end` stays low, or
  `x_factor_deg_prep_end`/`x_factor_deg_contact` near zero).
- Reversed or collapsed kinetic sequence (racket peaks at or before the hip/shoulder rotation peaks).
- Non-racket arm dropped or never raised (`non_racket_wrist_height_prep_end` at or below shoulder
  height).
- No weight transfer (`weight_transfer` = false) across most attempts.
- Inconsistent execution across attempts even when any single attempt looks fine
  (`cross_attempt_summary`'s `consistency_flag`).

## What this footage cannot tell you
- **Grip** is not verifiable from video. Only mention it if `racket_face_proxy_contact` strongly and
  consistently suggests the racket face is edge-on to the camera in a way inconsistent with a normal
  grip — and even then, hedge it heavily.
- Exact rotation degrees (see the 3D hedge above).
- Shuttle flight/outcome for shadow swings.
