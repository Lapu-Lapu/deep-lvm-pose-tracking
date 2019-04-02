#+TITLE: Pose Tracking Project

- recent pose tracking algorithms rely on DNNs for 2D Keypoint annotation
  in single frames
- they are furthermore model-free
- even though pretty robust, a problem is label switching
  after occlusion, especially with multiple people in one frame
- these problems can be solved by incorporating temporal information,
  for example using a GPDM for keypoint trajectories

- these problems would not arise with dynamical model-based approaches
- problem is high dimensionality of images