# Statistics appendix

Physical success is the environment/script terminal success before quality filtering. Accepted adds trajectory length, IK, clean contact, and insertion-quality gates.

For v1, Full minus Partial physical success was -2.5 percentage points over 320 paired states; paired bootstrap 95% interval was [-7.8, +2.8] points. For v2 it was -2.9 points over 450 paired states; paired bootstrap 95% interval was [-8.2, +2.2] points. Both intervals include zero.

Accepted-rate differences are descriptive because no paired interval was computed for that secondary gate. Failure-stage counts use the policy's recorded terminal failure reason. A single rollout contributes to one physical failure stage; quality-only rejections are counted separately and may fail more than one quality check.
