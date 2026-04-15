Sections:
Abstract
I Introduction
II Related Works
III Method
    III-A Optimal Control with Sampling-Based Optimization
    III-B Issues with existing SBMPC-based retargeting methods
    III-C Sampling-Based Trajectory Optimization
IV Evaluation
    IV-A Implementation details
    IV-B Performance evaluation
    IV-C Algorithm analysis
        IV-C 1 SBTO, an incremental warm-starting process
        IV-C 2 SBTO’s effective horizon
    IV-D Demonstration Augmentation
    IV-E Motion Tracking using Reinforcement Learning
V Conclusion
VI Limitations and Future Work
References

Files Content:

## Contents
- I Introduction
- II Related Works
- III Method
  - III-A Optimal Control with Sampling-Based Optimization
  - III-B Issues with existing SBMPC-based retargeting methods
  - III-C Sampling-Based Trajectory Optimization
- IV Evaluation
  - IV-A Implementation details
  - IV-B Performance evaluation
  - IV-C Algorithm analysis
    - IV-C 1 SBTO, an incremental warm-starting process
    - IV-C 2 SBTO’s effective horizon
  - IV-D Demonstration Augmentation
  - IV-E Motion Tracking using Reinforcement Learning
- V Conclusion
- VI Limitations and Future Work
- References

## Abstract

Abstract In this paper, we introduce DynaRetarget , a complete pipeline for retargeting human motions to humanoid control policies. The core component of DynaRetarget is a novel Sampling-Based Trajectory Optimization (SBTO) framework that refines imperfect kinematic trajectories into dynamically feasible motions. SBTO incrementally advances the optimization horizon, enabling optimization over the entire trajectory for long-horizon tasks.
We validate DynaRetarget by successfully retargeting hundreds of humanoid–object demonstrations and achieving higher success rates than the state of the art. The framework also generalizes across varying object properties, such as mass, size, and geometry, using the same tracking objective.
This ability to robustly retarget diverse demonstrations opens the door to generating large-scale synthetic datasets of humanoid loco-manipulation trajectories, addressing a major bottleneck in real-world data collection. A supplementary video
demonstrating the results is available here .

## I Introduction

Generating feasible loco-manipulation behaviors is a highly complex problem, as it requires handling the underactuation of both the robot and the manipulated object, as well as complex contact interactions. Traditional frameworks mostly relied either on gradient-based optimization to generate optimal trajectories [30], or on deep reinforcement learning (RL) [4] to directly learn optimal policies. The main advantage of trajectory optimization (TO) lies in its efficiency at finding locally optimal trajectories; however, it often requires augmentation with search-based methods to enable sufficient exploration [27, 21, 3]. On the other hand, while RL is very effective at generating robust behaviors, exploration remains a major challenge, frequently leading to heavy reward shaping for each individual motion.

One plausible remedy to the exploration problem is the use of demonstrations. In particular, thanks to the similarity between human and humanoid morphology, this line of research has recently attracted significant attention and has led to many impressive results [36, 16, 34]. The main idea behind these works is to retarget human motions to humanoid robots and then use the resulting trajectories as inputs to an RL policy, using domain randomization for sim-to-real transfer. In this way, the exploration problem is largely alleviated, and the RL component requires only a small set of simple reward terms that are shared across motions.

While the RL structure is largely standard among these approaches, the retargeting module differs substantially. Most methods solve a kinematic optimization problem for retargeting [34, 29, 6, 14], and are therefore susceptible to artifacts such as physical and geometric inconsistencies, particularly for loco-manipulation tasks. More recently, [16] proposed using sampling-based model predictive control (SBMPC) to improve retargeting quality. However, this approach solves only a short-horizon optimization problem at each iteration, making it difficult to handle long-horizon behaviors due to its inherently myopic nature.

In this paper, we introduce *DynaRetarget*, which combines inverse kinematic retargeting with a novel sampling-based trajectory optimization (SBTO) method that incrementally increases the optimization horizon to ultimately solve the original long-horizon problem (Fig. [1](https://arxiv.org/html/2602.06827v1#S0.F1)). The generated trajectories are then fed to an RL module to learn robust tracking policies using domain randomization during training. Through extensive ablation studies and comparisons, we show that *DynaRetarget* significantly improves success rates.

Figure: Figure 2: DynaRetarget overview. Given a human–object demonstration, we first perform IK-based retargeting to obtain a kinematically-feasible robot–object demonstration. Due to morphological differences between the human and the robot, this process can produce imperfections, for instance missing contacts (red circle). To address these issues, we use the kinematic trajectory as a reference for SBTO, which refines the trajectory and ensures its physical consistency, including removing missing contacts (green circle). The motion is then used to train an RL tracking policy in simulation with domain randomization. Finally, the learned policy is transferred zero-shot to our humanoid robot in the real world.
Refer to caption: https://arxiv.org/html/2602.06827/x2.png

The main contributions of this work are as follows:

- •
We introduce, to the best of our knowledge, the first sampling-based trajectory optimization method that retargets imperfect kinematic demonstrations into dynamically feasible humanoid loco-manipulation behaviors while considering the full problem horizon.
- •
We validate our approach in simulation on hundreds of motions, demonstrating significantly higher retargeting success rates than prior methods, and show that the resulting motions improve downstream RL policy learning and transfer robustly to a real humanoid robot across several loco-manipulation tasks.

## II Related Works

In computer graphics, motion retargeting has been extensively studied, demonstrating the effectiveness of data-driven approaches—particularly RL—for motion tracking in simulation across different morphologies using relatively simple reward structures [17, 18, 19]. Building on this line of work, [28, 32] extend motion imitation to loco-manipulation by incorporating human–object demonstrations and explicitly modeling object motion and contact information in the reward function. These methods have primarily been validated in simulation.

When transferring human motions to humanoid robots, additional challenges arise from significant morphological differences, such as disparities in degrees of freedom, limb lengths, and mass distributions. PHC [14], a retargeting method commonly used in robotics [6, 7], addresses this issue by selecting corresponding keypoints between the human and the humanoid robot and formulates an inverse kinematics (IK) problem based on these 3D keypoints. This is followed by an unconstrained optimization step to enforce physical consistency. However, the resulting motions often remain dynamically infeasible, exhibiting artifacts such as foot skating and penetrations. GMR [35] extends this approach by incorporating both keypoint positions and rotations into the IK formulation, but it still suffers from similar limitations.

More recently, [25] proposed a method that leverages text-to-image models to synthesize human–object interaction scenes, which are then retargeted via IK and used to warm-start a whole-body trajectory optimization. While this approach improves physical consistency, it requires extensive manual tuning of collision penalties for each robot–object interaction, limiting its scalability.

Several works have explored RL-based motion tracking for humanoid robots. For example, [13] introduces an RL-based framework with a reward structure similar to [17] and demonstrates the first successful deployment of highly dynamic locomotion policies on real hardware using retargeted human demonstrations. Extending this idea to loco-manipulation, [34] relies solely on proprioceptive observations and introduces a data augmentation technique to handle variations in object shape. Similarly, [29] uses monocular videos to recover global human motion via [24], which is then used as demonstration data to train RL policies with contact-based rewards for robot–object interaction.
Although RL has proven to be a robust and effective approach for generating dynamically feasible motions from retargeted kinematic trajectories, it typically requires long training times and high-quality demonstrations. For loco-manipulation, it also requires accurate contact information which has proven to improve the RL policy performance [36, 29], but are hard to accurately obtain from kinematic retargeting.

The work most closely related to ours is [16], which employs an SBMPC framework [8, 31, 33] to improve the geometric and dynamic consistency of retargeted motions. Given a kinematically retargeted trajectory, SBMPC generates dynamically consistent motions by repeatedly optimizing control trajectories over a short horizon in a receding horizon fashion.
However, because the optimization considers only a limited horizon at each step, the method is sensitive to imperfections in the demonstration. Failures can occur during the retargeting phase, as full-horizon consistency is not explicitly enforced.

## III Method

### III-A Optimal Control with Sampling-Based Optimization

Sampling-based (or zero-order) optimization algorithms address the generic optimization problem $\min_{x\in\mathbb{R}^{n}}f(x)$ by only point-wise evaluating a known function $f$ to be minimized.
The gradient of $f$ is not required, which makes it popular for non-smooth and non-convex optimization problems, such as optimal control of robots for contact-rich tasks.

In optimal control, the objective is to find a control sequence $\{\mathbf{u}_{0},\dots,\mathbf{u}_{T-1}\}$ minimizing a cost function $J$ while satisfying the system’s dynamics $\mathbf{x}_{t+1}=f_{dyn}(\mathbf{x}_{t},\mathbf{u}_{t})$, as formulated below:

$$ $\displaystyle\min_{\mathbf{u}_{0},\mathbf{u}_{1},\dots,\mathbf{u}_{T-1}}J(\mathbf{x}_{0:T},\mathbf{u}_{0:T-1})$ (1) $\displaystyle\text{s.t.}\quad\mathbf{x}_{0}=\mathbf{x}_{ini},\;\;\mathbf{x}_{t+1}=f_{\text{dyn}}(\mathbf{x}_{t},\mathbf{u}_{t}).$ $$

$\mathbf{x}_{t}=(\mathbf{q}_{t},\mathbf{v}_{t})\in\mathbb{R}^{n_{x}}$ and $\mathbf{u}_{t}\in\mathbb{R}^{n_{u}}$ denote respectively the state (joint positions and velocities) and control of the system at time $t$.
To satisfy the dynamics constraint, control sequences $\mathbf{u}_{0:T-1}$ are rolled-out in a single-shooting fashion from the initial state $\mathbf{x}_{0}$ using a simulator (treating it as a black box), which ultimately outputs state trajectories $\mathbf{x}_{0:T}$ needed to evaluate the cost $J(\mathbf{x}_{0:T},\mathbf{u}_{0:T-1})$.

The most commonly used algorithms in robotics are Cross-Entropy Method (CEM) [22], a special case of Covariance Matrix Adaptation (CMA) [5], and Model Predictive Path Integral (MPPI) [31]. In general, these approaches are used in a receding horizon fashion [20, 10]. These algorithms rely on similar mechanisms, namely trying to approximate gradients at each iteration [9]; however, they differ in how the sampling distribution is updated, as outlined in algorithm [1](https://arxiv.org/html/2602.06827v1#alg1).

To reduce the size of the sampling space, it is common to sample interpolation knots $\mathbf{k}\in\mathbb{R}^{K\cdot n_{u}}$ instead of the full control trajectory $\mathbf{u}_{0:T-1}$. Those knots are usually equally spread in time at steps $\boldsymbol{\tau}\in\mathbb{N}^{K}$, with $\tau_{0}=0$ and $\tau_{K-1}=T-1$.

Figure: Algorithm 1 FHTO, Sampling-Based Fixed Horizon Trajectory Optimization

### III-B Issues with existing SBMPC-based retargeting methods

Successful works in the literature that use zero-order optimization for retargeting [16, 12, 11] solve a short-horizon problem in the form of ([1](https://arxiv.org/html/2602.06827v1#S3.E1)) in an MPC fashion. However, such an approach suffers from three important issues:

- 1.
As MPC repeatedly solves a short-horizon problem, it can exhibit myopic behavior in long-horizon tasks. This effect is further exacerbated when using imperfect references that are physically inconsistent. For instance, if the contact geometry in the reference trajectory is inaccurate, the resulting short-horizon optimal plans may also fail to establish the correct contacts, ultimately leading to task failure.
- 2.
This is amplified as SBMPC-based retargeting behaves in a greedy fashion: once an action is executed, the system is simulated forward, and earlier actions cannot be re-optimized. For example, if SBMPC mistakenly drops the object during the early phases of a motion, recovery is highly unlikely, as doing so would require substantial deviation from the reference, which is penalized by the tracking cost.
- 3.
The trajectories produced by SBMPC tend to be jerky, as they are generated through feedback control. This lack of smoothness can negatively affect both the training process and the performance of downstream RL policies.

Considering the full horizon of the problem would avoid these pitfalls. However, humanoid loco-manipulation is a high-dimensional problem; optimizing all control variables simultaneously with a single-shooting sampling-based optimizer is therefore likely to fail, as both the number of variables and the number of local minima increase with the horizon length.
Our key observation is that the control variables toward the end of the trajectory strongly depend on those at the beginning. Updating the last control variables before the early ones are sufficiently optimized can lead to undesirable updates, making the optimization inefficient and potentially preventing convergence to desired behaviors. Based on this observation, in the next subsection, we introduce SBTO, a trajectory optimization framework that incrementally increases the optimization horizon.

### III-C Sampling-Based Trajectory Optimization

SBTO optimizes control variables incrementally. First, control knots $\mathbf{k}_{0}$ at time $\tau_{0}$ are optimized, then control knots $\{\mathbf{k}_{0},\mathbf{k}_{1}\}$ at time $\{\tau_{0},\tau_{1}\}$ (warm-starting with the previous solution $\mathbf{k}^{*}_{0}$), and so on, until all knots are being optimized.
The algorithm contains two nested loops: the outer loop incrementally increases the number of decision variables being optimized, while the inner loop repeatedly refines all the currently active variables. A skeleton of the algorithm can be seen in Algorithm [2](https://arxiv.org/html/2602.06827v1#alg2).

To optimize knots $\{\mathbf{k}_{0},\dots,\mathbf{k}_{k}\}$, we do not perform the full horizon roll-out until $T$, but a partial rollout until $\tau_{k}$, with $\tau_{k}$ corresponding to the time step of the last knot being optimized. Also, the optimization horizon $\tau_{k}$ grows incrementally.
At each increment $k$, this procedure is equivalent to solving the Fixed-Horizon Trajectory Optimization (FHTO) from Algorithm [1](https://arxiv.org/html/2602.06827v1#alg1) with truncated parameters $\boldsymbol{\mu}_{0:\kappa},\boldsymbol{\Sigma}_{0:\kappa,0:\kappa}$ and knot time $\boldsymbol{\tau}_{0:k}$. $\kappa=(k+1)n_{u}-1$ denotes the index of the last variable associated with knot $\mathbf{k}_{k}$.
Since $\tau_{0}=0$, the process starts at $k=1$ in practice.

Increments occur when the maximum diagonal value of the covariance matrix $\boldsymbol{\Sigma}$ is below a threshold $\sigma_{min}$, indicating that the optimization has sufficiently converged.
This adaptive criterion allows SBTO to adjust to the growing number of variables, as the convergence rate can change when more variables are being optimized.

Figure: Algorithm 2 SBTO

Intuitively, $\sigma_{min}$ plays an important role in the convergence of the algorithm.
Setting $\sigma_{min}$ large enough ensures that $\boldsymbol{\Sigma}_{0:\kappa,0:\kappa}$ has not fully converged before incrementing. This enables all variables in the current horizon window to be optimized, even after multiple increments, which prevents early convergence to local minima.
However, a too small $\sigma_{min}$ makes the sampling distribution shrink to a point distribution, which would prevent the first variables from escaping a potentially bad local minimum after incrementing.
Conversely, a too large $\sigma_{min}$ would make the increment happen too early with the variables too far from the optimum, which would almost be equivalent to solving the full-horizon TO from scratch. We verify this empirically in Section [IV-C2](https://arxiv.org/html/2602.06827v1#S4.SS3.SSS2).

Note that SBTO is tailored for problems having a dense cost $J$ where even a short-horizon window provides a meaningful estimate of the optimal solution. Retargeting tasks satisfy this property, which motivates our evaluation of the proposed method in this context.

## IV Evaluation

We evaluate SBTO on a trajectory refinement task using reference motions from the OmniRetarget dataset [34]. As input, we use kinematically retargeted trajectories from this dataset; in fact, SBTO performs dynamic refinement, correcting kinematically imperfect trajectories to produce dynamically feasible whole-body motions. The dataset contains hundreds of motions of a G1 humanoid robot interacting with a box, including pick-and-place, kicking, and pushing or dragging motions. Many of these trajectories exhibit missing contacts, penetrations, or discontinuities, making them challenging to refine.

In subsection [IV-A](https://arxiv.org/html/2602.06827v1#S4.SS1), we provide implementation details of SBTO. Subsection [IV-B](https://arxiv.org/html/2602.06827v1#S4.SS2) compares SBTO’s performance with a state-of-the-art SBMPC. In subsection [IV-C](https://arxiv.org/html/2602.06827v1#S4.SS3), we analyze the optimization process to highlight SBTO’s key properties. Section [IV-D](https://arxiv.org/html/2602.06827v1#S4.SS4) demonstrates that SBTO can adapt to objects with properties (shape, etc.) different from the original demonstrations. Finally, section [IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5) shows that the quality of SBTO’s output trajectories benefits the training and deployment of RL tracking policies.

### IV-A Implementation details

We implemented SBTO using the MuJoCo simulator [26] and leveraged its newly introduced rollout function to perform parallel rollouts on the CPU. We used a simulation timestep of $\Delta t=0.01$ s and considered the full collision model of the robot. The time interval between knots is independent of the reference and is set to $0.25$ s. The control sequence $\boldsymbol{u}_{0:T}$ corresponds to a PD target trajectory.

In all experiments, we use CEM [1] to update the sampling distribution. Following [20], we additionally retain a subset of elite samples across iterations ($N_{keep}=\lceil\rho_{k}\rho_{e}N\rceil$) and apply an exponentially weighted moving average (EWMA) with momentum parameters $\alpha_{\mu}$ and $\alpha_{\Sigma}$ to prevent premature shrinking of the distribution [2]. The initial mean of the distribution is set to the joint positions at each knot time step from the reference, $\boldsymbol{\mu}=\mathbf{q}^{ref}_{0:T}[\boldsymbol{\tau}]$, while the initial covariance is set to $\boldsymbol{\Sigma}=\sigma_{0}^{2}\mathbf{I}_{K\cdot n_{u}}$. We observed that considering the full covariance matrix improved convergence. Hyperparameter values are reported in Table [I](https://arxiv.org/html/2602.06827v1#S4.T1).

**TABLE I: CEM hyperparameters**
| Parameters | Value |
| --- | --- |
| Number of samples $N$ | 1024 |
| Elite set proportion $\rho_{e}$ | 0.03 |
| Keep elites proportion $\rho_{k}$ | 0.04 |
| Mean momentum $\alpha_{\mu}$ | 0.95 |
| Covariance momentum $\alpha_{\Sigma}$ | 0.2 |
| Initial std. $\sigma_{0}$ | 0.25 |

The cost function penalizes deviations in the state position $\mathbf{q}_{0:T}$ and velocity $\mathbf{v}_{0:T}$. Additional task-space terms enforce tracking of desired torso, foot, and hand poses. Contact-related terms discourage undesired collisions. The cost weights are summarized in Table [II](https://arxiv.org/html/2602.06827v1#S4.T2).

**TABLE II: Cost terms and corresponding weights. $\ominus{}$ denotes the subtraction between quaternions in the tangent space. Collision costs are computed as the number of collision events, which can be easily obtained with MuJoCo.**
| Cost term | Equation | Weight |
| --- | --- | --- |
| Motion Tracking |  |  |
| Joint position | $\|\mathbf{q}_{act}-\mathbf{q}_{act}^{\text{ref}}\|^{2}$ | 0.25 |
| Joint velocity | $\|\mathbf{v}_{act}-\mathbf{v}_{act}^{\text{ref}}\|^{2}$ | 0.01 |
| Base position | $\|{}^{W}\mathbf{p}_{\text{base}}-{}^{W}\mathbf{p}_{\text{base}}^{\text{ref}}\|^{2}$ | 5.0 |
| Base orientation | ${}^{W}\mathbf{q}_{\text{base}}\ominus{}^{W}\mathbf{q}_{\text{base}}^{\text{ref}}$ | 1.0 |
| Object position | $\|{}^{W}\mathbf{p}_{\text{object}}-{}^{W}\mathbf{p}_{\text{object}}^{\text{ref}}\|^{2}$ | 40.0 |
| Object orientation | ${}^{W}\mathbf{q}_{\text{object}}\ominus{}^{W}\mathbf{q}_{\text{object}}^{\text{ref}}$ | 4.0 |
| Object linear velocity | $\|{}^{W}\mathbf{v}_{\text{object}}-{}^{W}\mathbf{v}_{\text{object}}^{\text{ref}}\|^{2}$ | 0.2 |
| Motion Tracking (task space) |  |  |
| Torso position | $\|{}^{W}\mathbf{p}_{\text{torso}}-{}^{W}\mathbf{p}_{\text{torso}}^{\text{ref}}\|^{2}$ | 30.0 |
| Torso orientation | ${}^{W}\mathbf{q}_{\text{torso}}\ominus{}^{W}\mathbf{q}_{\text{torso}}^{\text{ref}}$ | 3.0 |
| Torso linear velocity | $\|{}^{W}\mathbf{v}_{\text{torso}}-{}^{W}\mathbf{v}_{\text{torso}}^{\text{ref}}\|^{2}$ | 0.3 |
| Torso angular velocity | $\|{}^{W}\mathbf{w}_{\text{torso}}-{}^{W}\mathbf{w}_{\text{torso}}^{\text{ref}}\|^{2}$ | 0.1 |
| Foot position | $\sum_{i\in\mathcal{F}}\|{}^{W}\mathbf{p}_{\text{foot}_{i}}-{}^{W}\mathbf{p}_{\text{foot}_{i}}^{\text{ref}}\|^{2}$ | 10.0 |
| Hand position | $\sum_{j\in\mathcal{H}}\|{}^{W}\mathbf{p}_{\text{hand}_{j}}-{}^{W}\mathbf{p}_{\text{hand}_{j}}^{\text{ref}}\|^{2}$ | 5.0 |
| Regularization |  |  |
| Robot–object collision | $\sum_{c\in\mathcal{C}_{\text{ro}}}1$ | 2.0 |
| Self-collision | $\sum_{c\in\mathcal{C}_{\text{self}}}1$ | 1.0 |

### IV-B Performance evaluation

We evaluate SBTO on all motions from the OmniRetarget dataset [34] that are shorter than $9$ s (285 motions in total).
SPIDER [16], a recently released SBMPC that achieves state-of-the-art results on many dynamic refinement tasks, serves as our baseline.
In SPIDER, the cost terms are based solely on the configurations $\boldsymbol{q}_{0:T}$ (and not their velocities). For a fair comparison, we also evaluate a variant of SBTO using a similar configuration - considering only the terms described in the first section of Table [II](https://arxiv.org/html/2602.06827v1#S4.T2) and omitting the velocity terms — referred to as SBTO_pos.
We compare the methods using three metrics: algorithm success rate, computational efficiency, and smoothness of the resulting trajectories (all described below). The results are reported in Table [III](https://arxiv.org/html/2602.06827v1#S4.T3).

We consider the refinement successful when the object trajectory has an average position error $E_{\mathrm{pos}}<10$cm and an average rotation error $E_{\mathrm{rot}}<25^{\circ}$. The error terms are defined as below:

$$ $\displaystyle E_{\mathrm{pos}}$ $\displaystyle=\frac{1}{T}\sum_{t=1}^{T}\left\|\mathbf{p}_{\mathrm{obj},t}-\mathbf{p}_{\mathrm{obj},t}^{\mathrm{ref}}\right\|_{2}$ (2) $\displaystyle E_{\mathrm{rot}}$ $\displaystyle=\frac{180}{\pi}\frac{1}{T}\sum_{t=1}^{T}\arccos\!\bigl(2\langle\mathbf{q}_{\mathrm{obj},t},\mathbf{q}_{\mathrm{obj},t}^{\mathrm{ref}}\rangle^{2}-1\bigr)$ (3) $$

We define computational efficiency $\eta_{\text{eff}}$ as the total number of simulation steps required in the optimization, divided by the duration of the reference. This makes the metric independent of the machine or simulator, providing a more meaningful measure than one based on compute time. For SBTO, the computational cost depends on the number of knots being optimized at iteration $i$, denoted by $k(i)$, as can be seen below:

$$ $\eta_{\text{eff}}\;=\;\frac{N_{\text{sim}}}{T\Delta t}\stackrel{{\scriptstyle SBTO}}{{=}}\frac{N}{T\Delta t}\sum_{i\in\mathcal{I}}\tau_{k(i)}$ (4) $$

Finally, we define the trajectory smoothness $S$ as the sum of accelerations of all actuated joints (obtained by finite differencing) over the full trajectory. For better interpretability, we normalize the result by the trajectory smoothness of its corresponding reference $\tilde{S}\;=\;\frac{S}{S_{\text{ref}}}$.

$$ $S\;=\;\sum_{t=2}^{T-1}\left\|\ddot{\mathbf{q}}_{t}\right\|_{1},\quad\text{with}\quad\ddot{\mathbf{q}}_{t}=\frac{\mathbf{q}_{t+1}-2\mathbf{q}_{t}+\mathbf{q}_{t-1}}{\Delta t^{2}}$ (5) $$

**TABLE III: Algorithm performance comparison. The computational efficiency and smoothness are averaged over the successful trajectories only. For the compute, we provide absolute and relative values (separated by —).**
| Algorithm | Success (%) $\uparrow$ | Smoothness $\downarrow$ | Compute $\eta_{\text{eff}}$ $\downarrow$ |
| --- | --- | --- | --- |
| SBTO | 74.6 | 1.7 | 405529 — 3.3 |
| SBTO_pos | 62.1 | 2.7 | 444924 — 3.6 |
| SPIDER | 37.9 | 3.4 | 123496 — 1. |

As summarized in Table [III](https://arxiv.org/html/2602.06827v1#S4.T3), SBTO outperforms SPIDER, achieving nearly twice the success rate and producing smoother refined trajectories. Even SBTO_pos shows a clear improvement over SPIDER, highlighting the algorithmic advantages of SBTO. This, however, comes at the cost of roughly three times more computation. The increased cost arises because SBTO requires rollouts from the initial state at each iteration, which entails a large number of simulation steps for the later increments.
In practice, the refinement process takes roughly $1$ minute per second of refined motion on a 112-core Intel(R) Xeon(R) Platinum 8480+ CPU.
Failure cases typically occur when references are of poor quality, especially if they include sudden changes in hand–object contact or abrupt flips in object orientation.

### IV-C Algorithm analysis

In this section, we analyze SBTO’s optimization process to highlight its core features. We identify two key properties that likely explain its superiority over both Fixed-Horizon TO (FHTO) and SBMPC.

- •
SBTO incrementally optimizes the controls, warm-starting larger-horizon problems from shorter ones up to the full horizon. This approach mitigates the convergence and instability issues observed in FHTO.
- •
SBTO optimizes decision variables over a horizon far longer than the SBMPC horizon, overcoming its inherent short-sightedness.

Figure: (a)

Figure: Figure 4: Evolution of the object position error at time $t^{0}$ during the optimization. The object position error steadily decreases for about $200$ iterations with SBTO. This shows that the first knots are still being optimized even after $10$ increments of the horizon, which corresponds to an effective horizon of around $3.4$ s (see vertical and horizontal red lines). Other baselines fails as the position error remains too high.
Refer to caption: https://arxiv.org/html/2602.06827/x9.png

To qualitatively highlight these properties, we considered a specific motion, i.e., sub_10_largebox_045.
In this $4.6$ s reference, a box is kicked forward at the very beginning of the trajectory. The box slides on the floor and finally stops at timestep $t^{0}=1$ s. Only the first two knots at $\tau_{0}$ and $\tau_{1}$ are responsible for the kicking motion.
Specifically, by tracking how the box position error at the fixed time $t^{0}$ evolves across optimization iterations, we can measure how long the first decision variables remain actively optimized. Since only the first knots influence the box motion at $t^{0}$, improvements in the box position at $t^{0}$ directly indicate that the initial control variables are being refined over an extended horizon.

The results are summarized in Fig. [4](https://arxiv.org/html/2602.06827v1#S4.F4) (averaged over $10$ seeds).
The top plot demonstrates how the horizon length $\tau_{k}$ grows over optimization iterations. Note that the number of iterations per increment may vary, as it depends on the rate at which the covariance $\mathbf{\Sigma}$ shrinks.

The bottom plot shows the box position error at time step $t^{0}$ as a function of the iterations. The position error is computed from the minimum cost state trajectory $\mathbf{x}_{0:\tau_{k}}^{*}$ at each iteration.
We compare SBTO to FHTO with different fixed horizon lengths.

For SBTO, the error can only be plotted when the growing horizon $\tau_{k}$ reaches $t^{0}=1$ s, as before that, the trajectory at $t^{0}$ is not even being produced by the roll-outs.
We note $i^{0}$ the first iteration at which $t^{0}=1s$ is within the optimization window (see green lines) and plot the error for SBTO starting from iteration $i^{0}$.
In contrast, for FHTO, we only consider horizons larger than $t^{0}$, therefore, the error can be computed for all optimization iterations.
Snapshot of the trajectories at $t^{0}=1$ s for all baselines can be seen in Fig. [3](https://arxiv.org/html/2602.06827v1#S4.F3).

#### IV-C 1 SBTO, an incremental warm-starting process

SBTO succeeded for all $10$ runs, whereas FHTO on the full motion length ($4.6$ s) systematically failed due to the robot falling and failing to kick the box correctly. This can be seen in Figure [3(d)](https://arxiv.org/html/2602.06827v1#S4.F3.sf4).
This shows two things. First, FHTO is unlikely to converge on such a complex contact-rich task.
Second, SBTO warm-starts the full-horizon problem efficiently, as once done incrementing, the only difference between SBTO and $4.6$ s FHTO is the state $(\boldsymbol{\mu},\boldsymbol{\Sigma})$ from which the sampling process starts.

#### IV-C 2 SBTO’s effective horizon

The box position error of SBTO decreases steadily until iteration $i^{1}\simeq 340$ (vertical red line), corresponding to a total optimization horizon of $t^{1}=3.4$ s (horizontal red line). This horizon is substantially longer than both the per-increment look-ahead increase ($0.25$ s) and the fixed horizon used in SPIDER ($1.2$ s). We refer to $t^{1}$ as the effective horizon of SBTO.

This behavior indicates that early control variables continue to be refined over many increments (approximately $10$), meaning that the first decision variables are optimized with a tracking objective evaluated over $3$ s of future motion.

To emphasize that optimizing over a longer-horizon is beneficial, we compare SBTO with a $t^{0}=1$ s FHTO, representative of an SBMPC-style setup. As one can see on the bottom plot of Figure [4](https://arxiv.org/html/2602.06827v1#S4.F4), FHTO fails as the final box position error remains above $10$cm. SPIDER fails for the same reason on this task.
Interestingly, the joint tracking with $1$ s FHTO seems satisfying, as one can see in Fig. [3(e)](https://arxiv.org/html/2602.06827v1#S4.F3.sf5). A plausible explanation for the failure could be that the object position cost is significant over such a short horizon.
In contrast, over longer horizons, the cumulative box position cost can only grow (since the box is not being moved after $t^{0}$), which effectively increases its weight in the objective.

By performing a parameter sweep over the two main hyperparameters impacting the convergence dynamics ($\sigma_{min}$, which controls when to increment, and $\alpha_{\Sigma}$, which controls the distribution shrinking rate), we show that the effective horizon is likely not an emergent property, but is primarily governed by $\sigma_{min}$. Results of the experiment are shown in Fig. [5](https://arxiv.org/html/2602.06827v1#S4.F5).

Figure: Figure 5: Effective horizon of SBTO for a parameter sweep over $\sigma_{min}$ and $\alpha_{\Sigma}$, averaged over $3$ runs. The effective horizon increases column by column, as $\sigma_{min}$ increases, whereas it stays almost identical for different $\alpha_{\Sigma}$ values.
Refer to caption: https://arxiv.org/html/2602.06827/x10.png

### IV-D Demonstration Augmentation

Figure: (a)

SBTO produces trajectories that deviates from the kinematic reference to ensure dynamic feasibility. One way to quantify how much it could deviate is to evaluate refinement performance under changes in object properties, such as mass, size, and geometry. This evaluation is also important in practice, as collecting new demonstrations is expensive.

We evaluate the success rate of SBTO on a box with different masses and sizes, and geometries. All experiments are based on a single motion reference (sub10_largebox_084_original). The original object size is a cubic box of length $0.31$cm and $0.6$kg mass. We use the same cost terms and optimization settings as in previous experiments.

SBTO successfully handled boxes with masses ranging from $0.1$ to $8$kg and sizes ranging from $0.2$m to $0.4$m.
Furthermore, the dynamic refinement process generalized beyond boxes and was also successful on a cylinder (diameter and height of $0.31$m), a chair, and a shelf (as can be seen on Figure [6](https://arxiv.org/html/2602.06827v1#S4.F6)). This shows that a single demonstration can be refined into dynamically feasible motions across a diverse set of object geometries and physical properties.

### IV-E Motion Tracking using Reinforcement Learning

With access to physically consistent trajectories for humanoid-object loco-manipulation, we conduct extensive experiments on training RL tracking controllers using PPO [23], similar to [13], while using a residual action space as in [29]. We use additional observations to track the object trajectory throughout the episode. In addition to the one-step desired robot trajectory, the policy observes the one-step object pose error with respect to the desired pose, as well as the object pose expressed in the robot frame. As an additional objective to our controller, we also task the policy to track the object pose. We use contact rewards to incentivize contact for the different end-effectors with the object. The contact reward also gradually penalizing forces that exceed $10$N. Since we have access to physically consistent trajectories using our method, we can naturally extract accurate contacts from the same simulator, without relying on any heuristics. For all of our experiments, we used the same set of reward weights as illustrated in Table [IV](https://arxiv.org/html/2602.06827v1#S4.T4).

**TABLE IV: Reward terms used by the RL tracking controller.**
| Reward Term | Equation | Weight |
| --- | --- | --- |
| Motion Tracking |  |  |
| Root Position | $\exp\!\left(-5\|\mathbf{p}_{t}-\mathbf{p}_{t}^{\text{ref}}\|^{2}\right)$ | 0.5 |
| Root Orientation | $\exp\!\left(-3\|\mathbf{q}_{t}-\mathbf{q}_{t}^{\text{ref}}\|^{2}\right)$ | 0.5 |
| Body Position | $\exp\!\left(-5\|\mathbf{p}_{\text{body},t}-\mathbf{p}_{\text{body},t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Body Orientation | $\exp\!\left(-3\|\boldsymbol{\theta}_{\text{body},t}-\boldsymbol{\theta}_{\text{body},t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Body Linear Velocity | $\exp\!\left(-0.5\|\mathbf{v}_{t}-\mathbf{v}_{t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Body Angular Velocity | $\exp\!\left(-0.05\|\boldsymbol{\omega}_{t}-\boldsymbol{\omega}_{t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Joint Pos Tracking | $\exp\!\left(-5\|\boldsymbol{u}_{t}-\boldsymbol{u}_{t}^{\text{ref}}\|^{2}\right)$ | 2.0 |
| Object Tracking |  |  |
| Contact Match | $\mathbb{I}(c=c^{\text{ref}})\,\cdot\exp\!\left(-0.1\left(\left\|\boldsymbol{F}_{t}\right\|-10\right)\right)$ | 1.25 |
| Object Position | $\exp\!\left(-8\|\mathbf{p}_{\text{obj},t}-\mathbf{p}_{\text{obj},t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Object Orientation | $\exp\!\left(-5\|\boldsymbol{\theta}_{\text{obj},t}-\boldsymbol{\theta}_{\text{obj},t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Object Linear Velocity | $\exp\!\left(-2\|\mathbf{v}_{\text{obj},t}-\mathbf{v}_{\text{obj},t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Object Angular Velocity | $\exp\!\left(-0.2\|\boldsymbol{\omega}_{\text{obj},t}-\boldsymbol{\omega}_{\text{obj},t}^{\text{ref}}\|^{2}\right)$ | 1.0 |
| Regularization |  |  |
| Action Rate | $-\|\mathbf{a}_{t}-\mathbf{a}_{t-1}\|^{2}$ | $-0.1$ |
| Joint Limit | $-\sum_{i}\phi(q_{i},q_{i,\min},q_{i,\max})$ | $-10.0$ |
| Self-collisions | $\sum_{c\in\mathcal{C}_{\text{self}}}1$ | $-1.0$ |

**TABLE V: Downstream RL policy evaluation using different references to track**
| Method | Success Rate $\uparrow$ | MPKPE $\downarrow$ | Object Pos Error $\downarrow$ | Object Ori Error $\downarrow$ |
| --- | --- | --- | --- | --- |
|  | (%) | (cm) | (cm) | (rad) |
| OmniRetarget [34] | $79.41\pm 32.57$ | $3.67\pm 0.33$ | $12.50\pm 6.70$ | $0.18\pm 0.09$ |
| DynaRetarget (Ours) | $\mathbf{97.09\pm 2.31}$ | $\mathbf{3.57\pm 0.46}$ | $\mathbf{8.81\pm 1.16}$ | $\mathbf{0.11\pm 0.02}$ |

To enable robust transfer of our loco-manipulation policies to the real robot, we employ additional domain randomization techniques apart from those used in [13] upon resetting the episode with adaptive sampling, which samples more difficult portions of the trajectory more frequently than others. Specifically, when the episode is reset during the initial frames, we randomize the object pose $(\mathbf{p}_{\text{obj},t},\boldsymbol{\theta}_{\text{obj},t})$ around the desired object pose $(\mathbf{p}_{\text{obj},t}^{\text{ref}},\boldsymbol{\theta}_{\text{obj},t}^{\text{ref}})$. When the reset occurs after the initial frames, we instead randomize the object velocities $(\mathbf{v}_{\text{obj},t},\boldsymbol{\omega}_{\text{obj},t})$ around their desired values $(\mathbf{v}_{\text{obj},t}^{\text{ref}},\boldsymbol{\omega}_{\text{obj},t}^{\text{ref}})$. In addition, we apply random external pushes to the object at random intervals during the episode, randomize the object friction parameters, and vary its mass around the nominal value. Crucially, we do not introduce any additional termination conditions related to object tracking, as we found that such terminations consistently degrade performance and exacerbate the learning difficulty when combined with adaptive sampling during training from scratch.

We use mjlab, which uses the GPU-optimized version of MuJoCo, with IsaacLab-style [15] API design for training our motion tracking controllers with $8192$ envs for $10000$ iterations on a single NVIDIA RTX 4090 GPU.

To highlight the importance of physical consistency of the trajectories for downstream RL policy performance, we compare success rates and tracking metrics against policies trained with trajectories from OmniRetarget [34] in Table [V](https://arxiv.org/html/2602.06827v1#S4.T5). We left out comparisons with SBMPC-based methods since they do not succeed on diverse enough motions for reasons explicated in Section [III-B](https://arxiv.org/html/2602.06827v1#S3.SS2) and evaluated in Table [III](https://arxiv.org/html/2602.06827v1#S4.T3). Results are averaged over $1024$ episodes spanning $8$ distinct motions and diverse initial configurations. These motions cover a broad range of motions such as lifting, pushing with hands, pushing with legs etc. A policy rollout is deemed successful if the object pose remains within a predefined threshold of the desired object pose at every timestep of the trajectory, since the policy may continue to imitate the robot motion even when object tracking fails in the case of infeasible trajectories. In contrast to prior approaches that rely on artificial curriculum to mitigate such artifacts [36], our method enables the RL tracking controller to reliably learn even the most challenging behaviors such as object sliding and object manipulation using the robot’s legs without additional curriculum shaping as show in Figure [1](https://arxiv.org/html/2602.06827v1#S0.F1).

A major advantage of our method is its superior sample efficiency, as shown in Figure [7](https://arxiv.org/html/2602.06827v1#S4.F7). RL tracking policies converge significantly faster to the desired solution when trained on perfectly dynamically consistent trajectories from SBTO using the same simulator (MuJoCo), without requiring additional tuning. In contrast, policies trained on kinematically retargeted data alone either take substantially longer to converge or fail to track the object entirely, as joints have to deviate more from the reference to close the dynamic feasibility gap.

Figure: (a)

## V Conclusion

In this work, we presented *DynaRetarget*, a complete pipeline for transferring human motion to real-world deployed control policies. The central contribution of this pipeline is SBTO, a sampling-based trajectory optimization framework that refines imperfect kinematic humanoid trajectories into dynamically feasible motions.

SBTO incrementally grows the optimization horizon, effectively warm-starting the full-horizon problem while still allowing early decision variables to be refined as the horizon grows. This strategy mitigates the convergence challenges of sampling-based methods on long-horizon high-dimensional problems, while simultaneously overcoming the short-sightedness of SBMPC.

We extensively evaluated SBTO on hundreds of motions and showed that it achieves a substantially higher success rate than a state-of-the-art SBMPC baseline, while producing smoother trajectories. We further demonstrated that SBTO-generated trajectories benefit downstream learning: RL tracking controllers trained on them acquire more reliable object interaction behaviors without additional curriculum shaping and transfer successfully to real hardware.
Finally, SBTO generalizes across variations in object mass, size, and geometry for the same reference motion, highlighting its potential as a scalable approach for generating large-scale synthetic datasets of dynamically consistent humanoid loco-manipulation trajectories.

## VI Limitations and Future Work

The main limitation of SBTO lies in its scalability with respect to the duration of the trajectory. Unlike SBMPC, which operates over a fixed horizon, SBTO performs rollouts over an increasing number of steps, which can become computationally demanding for refining longer motions. Addressing this limitation is a key direction for future work. SBTO is also currently limited to tasks that provide a dense optimization cost.
One potential way to further improve scalability is to employ a multi-modal sampling distribution instead of the current multivariate gaussian. This would enable the simultaneous optimization of multiple candidate trajectories within a single optimization process, potentially reducing the computational cost per refined trajectory.
Another promising future direction is to use SBTO to track human keypoints directly, bypassing explicit kinematic retargeting.

## References

- [1]
P. Boer, D. Kroese, S. Mannor, and R. Rubinstein (2005-02)
A tutorial on the cross-entropy method.
Annals of Operations Research 134, pp. 19–67.
External Links: [Document](https://dx.doi.org/10.1007/s10479-005-5724-z)
Cited by: [§IV-A](https://arxiv.org/html/2602.06827v1#S4.SS1.p2.5).
- [2]
Z. I. Botev, D. P. Kroese, R. Y. Rubinstein, and P. L’Ecuyer (2013)
Chapter 3 - the cross-entropy method for optimization.
In Handbook of Statistics, C.R. Rao and V. Govindaraju (Eds.),
Handbook of Statistics, Vol. 31, pp. 35–59.
External Links: ISSN 0169-7161,
[Document](https://dx.doi.org/https%3A//doi.org/10.1016/B978-0-444-53859-8.00003-5),
[Link](https://www.sciencedirect.com/science/article/pii/B9780444538598000035)
Cited by: [§IV-A](https://arxiv.org/html/2602.06827v1#S4.SS1.p2.5).
- [3]
M. Ciebielski, V. Dhédin, and M. Khadiv (2025)
Task and motion planning for humanoid loco-manipulation.
In 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids),
pp. 1179–1186.
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p1.1).
- [4]
S. Ha, J. Lee, M. van de Panne, Z. Xie, W. Yu, and M. Khadiv (2025)
Learning-based legged locomotion: state of the art and future perspectives.
The International Journal of Robotics Research 44 (8), pp. 1396–1427.
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p1.1).
- [5]
N. Hansen and A. Ostermeier (2001)
Completely derandomized self-adaptation in evolution strategies.
Evolutionary computation 9 (2), pp. 159–195.
Cited by: [§III-A](https://arxiv.org/html/2602.06827v1#S3.SS1.p5.1).
- [6]
T. He, J. Gao, W. Xiao, Y. Zhang, Z. Wang, J. Wang, Z. Luo, G. He, N. Sobanbab, C. Pan, Z. Yi, G. Qu, K. Kitani, J. Hodgins, L. ”. Fan, Y. Zhu, C. Liu, and G. Shi (2025)
ASAP: aligning simulation and real-world physics for learning agile humanoid whole-body skills.
External Links: 2502.01143,
[Link](https://arxiv.org/abs/2502.01143)
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p3.1),
[§II](https://arxiv.org/html/2602.06827v1#S2.p2.1).
- [7]
T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. Kitani, C. Liu, and G. Shi (2024)
OmniH2O: universal and dexterous human-to-humanoid whole-body teleoperation and learning.
External Links: 2406.08858,
[Link](https://arxiv.org/abs/2406.08858)
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p2.1).
- [8]
T. Howell, N. Gileadi, S. Tunyasuvunakool, K. Zakka, T. Erez, and Y. Tassa (2022)
Predictive sampling: real-time behaviour synthesis with mujoco.
arXiv preprint arXiv:2212.00541.
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p5.1).
- [9]
A. Jordana, J. Zhang, J. Amigo, and L. Righetti (2025)
An introduction to zero-order optimization techniques for robotics.
External Links: 2506.22087,
[Link](https://arxiv.org/abs/2506.22087)
Cited by: [§III-A](https://arxiv.org/html/2602.06827v1#S3.SS1.p5.1).
- [10]
V. Kurtz and J. W. Burdick (2025)
Generative predictive control: flow matching policies for dynamic and difficult-to-demonstrate tasks.
arXiv preprint arXiv:2502.13406.
Cited by: [§III-A](https://arxiv.org/html/2602.06827v1#S3.SS1.p5.1).
- [11]
V. Kurtz (2024)
Hydrax: sampling-based model predictive control on gpu with jax and mujoco mjx.
Note: https://github.com/vincekurtz/hydrax
Cited by: [§III-B](https://arxiv.org/html/2602.06827v1#S3.SS2.p1.1).
- [12]
A. T. Le, K. Nguyen, M. N. Vu, J. Carvalho, and J. Peters (2025)
Model tensor planning.
External Links: 2505.01059,
[Link](https://arxiv.org/abs/2505.01059)
Cited by: [§III-B](https://arxiv.org/html/2602.06827v1#S3.SS2.p1.1).
- [13]
Q. Liao, T. E. Truong, X. Huang, Y. Gao, G. Tevet, K. Sreenath, and C. K. Liu (2025)
BeyondMimic: from motion tracking to versatile humanoid control via guided diffusion.
External Links: 2508.08241,
[Link](https://arxiv.org/abs/2508.08241)
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p4.1),
[§IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5.p1.1),
[§IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5.p2.4).
- [14]
Z. Luo, J. Cao, A. Winkler, K. Kitani, and W. Xu (2023)
Perpetual humanoid control for real-time simulated avatars.
External Links: 2305.06456,
[Link](https://arxiv.org/abs/2305.06456)
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p3.1),
[§II](https://arxiv.org/html/2602.06827v1#S2.p2.1).
- [15]
M. Mittal, P. Roth, J. Tigue, and et. al. (2025)
Isaac lab: a gpu-accelerated simulation framework for multi-modal robot learning.
arXiv preprint arXiv:2511.04831.
External Links: [Link](https://arxiv.org/abs/2511.04831)
Cited by: [§IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5.p3.2).
- [16]
C. Pan, C. Wang, H. Qi, Z. Liu, H. Bharadhwaj, A. Sharma, T. Wu, G. Shi, J. Malik, and F. Hogan (2025)
SPIDER: scalable physics-informed dexterous retargeting.
External Links: 2511.09484,
[Link](https://arxiv.org/abs/2511.09484)
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p2.1),
[§I](https://arxiv.org/html/2602.06827v1#S1.p3.1),
[§II](https://arxiv.org/html/2602.06827v1#S2.p5.1),
[§III-B](https://arxiv.org/html/2602.06827v1#S3.SS2.p1.1),
[§IV-B](https://arxiv.org/html/2602.06827v1#S4.SS2.p1.2).
- [17]
X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne (2018)
Deepmimic: example-guided deep reinforcement learning of physics-based character skills.
ACM Transactions On Graphics (TOG) 37 (4), pp. 1–14.
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p1.1),
[§II](https://arxiv.org/html/2602.06827v1#S2.p4.1).
- [18]
X. B. Peng, E. Coumans, T. Zhang, T. Lee, J. Tan, and S. Levine (2020)
Learning agile robotic locomotion skills by imitating animals.
arXiv preprint arXiv:2004.00784.
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p1.1).
- [19]
X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa (2021)
Amp: adversarial motion priors for stylized physics-based character control.
ACM Transactions on Graphics (TOG) 40 (4), pp. 1–20.
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p1.1).
- [20]
C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek, and G. Martius (2020)
Sample-efficient cross-entropy method for real-time planning.
External Links: 2008.06389,
[Link](https://arxiv.org/abs/2008.06389)
Cited by: [§III-A](https://arxiv.org/html/2602.06827v1#S3.SS1.p5.1),
[§IV-A](https://arxiv.org/html/2602.06827v1#S4.SS1.p2.5).
- [21]
B. Ponton, M. Khadiv, A. Meduri, and L. Righetti (2021)
Efficient multicontact pattern generation with sequential convex approximations of the centroidal dynamics.
IEEE Transactions on Robotics 37 (5), pp. 1661–1679.
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p1.1).
- [22]
R. Y. Rubinstein and D. P. Kroese (2004)
The cross-entropy method: a unified approach to combinatorial optimization, monte-carlo simulation and machine learning.
Springer Science & Business Media.
Cited by: [§III-A](https://arxiv.org/html/2602.06827v1#S3.SS1.p5.1).
- [23]
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)
Proximal policy optimization algorithms.
External Links: 1707.06347,
[Link](https://arxiv.org/abs/1707.06347)
Cited by: [§IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5.p1.1).
- [24]
Z. Shen, H. Pi, Y. Xia, Z. Cen, S. Peng, Z. Hu, H. Bao, R. Hu, and X. Zhou (2024-12)
World-grounded human motion recovery via gravity-view coordinates.
In SIGGRAPH Asia 2024 Conference Papers,
SA ’24, pp. 1–11.
External Links: [Link](http://dx.doi.org/10.1145/3680528.3687565),
[Document](https://dx.doi.org/10.1145/3680528.3687565)
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p4.1).
- [25]
I. Taouil, H. Zhao, A. Dai, and M. Khadiv (2025)
Physically consistent humanoid loco-manipulation using latent diffusion models.
In 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids),
pp. 1179–1186.
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p3.1).
- [26]
E. Todorov, T. Erez, and Y. Tassa (2012)
Mujoco: a physics engine for model-based control.
In 2012 IEEE/RSJ international conference on intelligent robots and systems,
pp. 5026–5033.
Cited by: [§IV-A](https://arxiv.org/html/2602.06827v1#S4.SS1.p1.3).
- [27]
M. A. Toussaint, K. R. Allen, K. A. Smith, and J. B. Tenenbaum (2018)
Differentiable physics and stable modes for tool-use and manipulation planning.
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p1.1).
- [28]
Y. Wang, J. Lin, A. Zeng, Z. Luo, J. Zhang, and L. Zhang (2023)
PhysHOI: physics-based imitation of dynamic human-object interaction.
External Links: 2312.04393,
[Link](https://arxiv.org/abs/2312.04393)
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p1.1).
- [29]
H. Weng, Y. Li, N. Sobanbabu, Z. Wang, Z. Luo, T. He, D. Ramanan, and G. Shi (2025)
HDMI: learning interactive humanoid whole-body control from human videos.
External Links: 2509.16757,
[Link](https://arxiv.org/abs/2509.16757)
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p3.1),
[§II](https://arxiv.org/html/2602.06827v1#S2.p4.1),
[§IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5.p1.1).
- [30]
P. M. Wensing, M. Posa, Y. Hu, A. Escande, N. Mansard, and A. D. Prete (2024)
Optimization-based control for dynamic legged robots.
IEEE Transactions on Robotics 40 (), pp. 43–63.
External Links: [Document](https://dx.doi.org/10.1109/TRO.2023.3324580)
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p1.1).
- [31]
G. Williams, A. Aldrich, and E. A. Theodorou (2017)
Model predictive path integral control: from theory to parallel computation.
Journal of Guidance, Control, and Dynamics 40 (2), pp. 344–357.
External Links: [Document](https://dx.doi.org/10.2514/1.G001921),
[Link](https://doi.org/10.2514/1.G001921),
https://doi.org/10.2514/1.G001921
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p5.1),
[§III-A](https://arxiv.org/html/2602.06827v1#S3.SS1.p5.1).
- [32]
S. Xu, H. Y. Ling, Y. Wang, and L. Gui (2025)
InterMimic: towards universal whole-body control for physics-based human-object interactions.
External Links: 2502.20390,
[Link](https://arxiv.org/abs/2502.20390)
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p1.1).
- [33]
H. Xue, C. Pan, Z. Yi, G. Qu, and G. Shi (2024)
Full-order sampling-based mpc for torque-level locomotion control via diffusion-style annealing.
External Links: 2409.15610,
[Link](https://arxiv.org/abs/2409.15610)
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p5.1).
- [34]
L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, and G. Shi (2025)
OmniRetarget: interaction-preserving data generation for humanoid whole-body loco-manipulation and scene interaction.
External Links: 2509.26633,
[Link](https://arxiv.org/abs/2509.26633)
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p2.1),
[§I](https://arxiv.org/html/2602.06827v1#S1.p3.1),
[§II](https://arxiv.org/html/2602.06827v1#S2.p4.1),
[§IV-B](https://arxiv.org/html/2602.06827v1#S4.SS2.p1.2),
[§IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5.p4.2),
[TABLE V](https://arxiv.org/html/2602.06827v1#S4.T5.8.8.5),
[§IV](https://arxiv.org/html/2602.06827v1#S4.p1.1).
- [35]
GMR: general motion retargeting
Note: GitHub repository
External Links: [Link](https://github.com/YanjieZe/GMR)
Cited by: [§II](https://arxiv.org/html/2602.06827v1#S2.p2.1).
- [36]
S. Zhao, Y. Ze, Y. Wang, C. K. Liu, P. Abbeel, G. Shi, and R. Duan (2025)
ResMimic: from general motion tracking to humanoid whole-body loco-manipulation via residual learning.
External Links: 2510.05070,
[Link](https://arxiv.org/abs/2510.05070)
Cited by: [§I](https://arxiv.org/html/2602.06827v1#S1.p2.1),
[§II](https://arxiv.org/html/2602.06827v1#S2.p4.1),
[§IV-E](https://arxiv.org/html/2602.06827v1#S4.SS5.p4.2).