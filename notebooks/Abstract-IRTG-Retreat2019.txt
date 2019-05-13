### Abstract

Pose tracking is the estimation of skeletal poses from video frames. We
investigate pose tracking as a component of biological movement perception. For
this purpose it is important to incorporate temporal and physical constraints
of biological movement. Most state of the art pose tracking algorithms predict
2D keypoint locations from single frames instead, and are thus not suitable for
this purpose.

We assume that movement represented as skeletal joint angles share a latent
representation with its video representation. We implement this assumption as
deep latent variable model (also known as variational autoencoder). The latent
space can be constrained by dynamical systems. Pose tracking in this framework
is done via conditioning the video frames.

We demonstrate the feasibility of the approach using toy data. This is an
important step towards a robust and principled probabilistic pose tracking
algorithm for real world data. The model additionaly allows for the generation
of videos given poses. It is thus an ideal test bed for hypotheses about the
perception of biological movement.
