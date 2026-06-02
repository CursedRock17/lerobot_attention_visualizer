# GROOT
This document should provide you with the necessary context to parse the various GR00T foundation models, particularly the N1.5 version.

## Paper
The following link will provide a PDF copy of the GROOT-N1 paper: https://arxiv.org/pdf/2503.14734 
use !`\WebFetch` to read the paper and import context about why certain decisions were made in context of the paper. There's an image of the model architecture (Figure 3) in section 2.1 which should highlight the overall model architecture. 
The model architecture should help drive information about the underlying Eagle 2 visual transformer. If need be, you can get extra context from !`git clone https://github.com/NVlabs/Eagle/tree/main/Eagle2_5` which is the repository from the Eagle2.5 rollout. More importantly is the Eagle-2 README: https://github.com/NVlabs/Eagle/blob/main/Eagle/README.md
The actual model architecture can be found at !`git clone https://github.com/NVIDIA/Isaac-GR00T -b n1.5-release` which houses the N1.5 NVIDIA model of GROOT, this is a raw python rollout, so it should stil be useful. Huggingf ace also provides a card that you can use: https://huggingface.co/nvidia/GR00T-N1.5-3B
The **MOST** important link that you'll have access to is the underlying GROOT rollout from LeRobot, in which they implement the GROOT policy. The following weblink can also be access in the local repository with a similar path: https://github.com/huggingface/lerobot/tree/v0.4.3/src/lerobot/policies/groot. There is an entire dedicated "action_head" directory which provides the "action_encoder", "cross_attention_dit", and "flow_matching_action_head" python files.

## Writing the policy
It's import that we can easily extend this library to user of Huggingface's LeRobot library. There's a local version @lerobot linked in this directory, but using the !`\lerobot-context` command will provide you information about lerobot as a whole. The current library already allows easy rollout of the policy to live running examples and past examples on datasets which makes it easy to analyze different parts of data collection. 
We have the ability to extract higher speeds with PyTorch and JIT capabilites. Take a look at the @docs/ directory for gaining more information about the current setup.
