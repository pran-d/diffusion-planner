# Changes in the abstract:
- don't mention physical constraints. Use less strict words like kinematically consistent. The output of our diffusion model is not dynamically feasible, but is NEARLY dynamically feasible which can be handled by our pipeline containing a RL tracker.
- don't say "final system integrates". in this thesis, we explore these different robustness mechanisms such as CFG, in-betweening. also, the physics-aware runtime "safeguards" are just heuristic, they don't ensure safety. don't mention fused min-SNR loss reweighting here, this is merely copied from another work and I haven't done any contribution in that.
- Don't mention what metrics are used to evaluate the reconstruction and generalization. Don't mention exact endpoint convergence and strict physical validity in the abstract. Explain this as trade-offs in the main content of the thesis.
- The purpose of this thesis is to evaluate diffusion models for kinematic trajectory generation in a broader pipeline that uses a RL tracker to follow these kinematic plans. Don't mention the last part about arguing for hybrid pipelines etc.


# Changes in the introduction:
- add more content from iitm-report/ here. you need to motivate the need for imitation learning first, and then explain why learning from sparse demonstrations is needed.
- give importance to data representations that reduce the sparsity of demonstrations (egocentric, segmented, etc.)
- explain the need to break the correlation between different features, which is done through masking during diffusion training (get some info from codebase while writing this)
- avoid using the word waypoint
- explain the use of separate kinematic planner and dynamic controller in order for real-time control, slow inference-time of large generative models. 


# Changes in the problem formulation:
- Include kinematic motion generation as part of the problem formulation, as shown in the mid-term report and in ![this image](image.png)
- Explain that diverse combinations of different body part motion sequences are required in order to achieve the full task-space. Instead of learning the entire motion demonstration, we want to learn motion primitives from the trajectory that can be reused to generate new trajectories.
- The final conceptual objective in constrained planning view should be a separate paragraph
- In the intro paragraph, explain that there are multiple ways of looking at this problem each dealing with different aspects. Maintain research tone.
- Don't mention the fused min-SNR stuff. 
- Sections 2.4, 2.5 and 2.6 can be combined into one section titled Conditional Diffusion for Long-Horizon Trajectory Planning

- Add the fact that the actual task would include object linear displacement and rotation (total 6D), but for simplicity, we used only a 3D task space. 
- Avoid repeating the same things again.

- Add some context for the dataset: our expert demonstrations come from a pipeline involving kinematic retargeting from human videos, followed by sampling-based trajectory optimization to obtain dynamically feasible trajectories. For this, cite omniretarget and dynaretarget, as the source of our data is dynaretarget
- Explain what v-prediction means, in context of diffusion.

# Change in problem formulation and motivation
- Add data augmentation as part of the problem formulation and motivation: using the diffusion model and a RL tracker in a evolutionary pipeline (refer to kanish-midterm-thesis for this, instead of the MLP as motion generator we are now using the diffusion)


# Change in related work
- Pull points from most related papers in references/
- Change the structure of this section to flow more naturally, don't have a separate section for insights from recent large-scale and physics-coupled systems. 
- Instead of naming papers like CLoSD, UniPhys, BeyondMimic and so on, cite them using \cite  


# Change in methodology and problem formulation
- Don't include the point 3 in section 4.1. For point 4, add composing part-wise sequences from different trajectories to generate new trajectories.
- Remove sections 2.5.1 and 2.5.2 from problem formulation, they should instead be a part of the methodology (section 4.2)
- Don't have the titles like Representation-level contributions and Architecture-level constributions. This should read as a research paper, write accordingly. 
- Have data augmentation (4.3.3) as a separate subsection which explains mirror symmetry, and any other data operations i use in the code such as goal vector normalization, goal magnitude clipping, etc.
- For architecture, include the image of the network architecture as well. 
- Use a mathematical equation to explain repaint-style resampling.  
- Explain indicator-aware conditioning as part of the waypoint-based in-betweening
- Remove section 4.6, 4.8 from here for now, instead add a section called ablations with all the config parameters
- Motivate the need for physics-based auxiliary losses. Explain the auxiliary losses using math expressions as well. Don't include feet-sliding proxy loss.


# Add a separate chapter for the evolutionary pipeline and the RL controller
- Include text and figures from Kanish's midterm thesis report.
- Note that the motion generator is now my diffusion model. You can analyze the code in kanish-midterm-thesis/mjlab/ to understand the pipeline.


# Next changes in all files:
- Remove section 4.6 about ablations and configurable factors
- In section 5.2.4, don't explain the gram-schmidt orthogonalization. Instead explain how to get the 6D continuous representation from the rotation matrix.
- Check the ablation results from reports/ablation_eval/ and write it into the report for 06_results.tex. Make sure that the ablations match with observations.md. Add images from these ablations into the report.
- Don't mention "midterm report" and things like that in the thesis. The entire report should read like a research paper
- Add more content in the RL tracking controller and evolutionary pipeline, including some results for task space coverage.