# robotics Evaluation Metrics

## 1.2 Metrics Categories

### 1.2.1 Task Performance Metrics

#### Success Rate (SR)
Success Rate [1] serves as the fundamental metric for robotics evaluation, measuring the percentage of successfully completed tasks. Research shows significant variation in success rates across different robotics policies and tasks:

$$SR = \frac{N_{\text{success}}}{N_{\text{total}}}$$

Current models achieve 12.4% average success on pick-up tasks, 6.0% on move-near tasks, 1.2% on put-on tasks, and 0.5% on put-in tasks. The RT-1-400k model demonstrates the highest performance with 34.4% success on pick-up tasks.

#### Task Completion Rate
This evaluates the ability to execute multi-step task sequences, revealing critical limitations in current robotics policies[1]. Analysis shows that success rates drop significantly between sequential steps, indicating challenges in interpreting complex natural language instructions that require multiple actions.

$$TCR = \frac{N_{\text{completed tasks}}}{N_{\text{task chains}}}$$

#### Action Accuracy
This measures precision of predicted actions against ground truth trajectories, typically evaluated using Mean Squared Error (MSE) and its variations including Average MSE(AMSE) and Normalized AMSE (NAMSE)[2][3]. This metric provides direct assessment of model performance in offline evaluation scenarios where physical robot deployment is not feasible.

$$MSE = \frac{1}{T}\sum_{t=1}^{T} \|\mathbf{a}_t - \hat{\mathbf{a}}_t\|_2^2$$

$$AMSE = \frac{1}{K}\sum_{k=1}^{K} MSE_k$$

$$NAMSE = \frac{AMSE}{\sigma^2_{\text{action}}}$$

### 1.2.2 Trajectory Quality Metrics

#### Path Length
Path Length quantifies the total distance traveled to complete a task[4], serving as a crucial metric for efficiency evaluation. The metric can be calculated for both actual robot trajectories and virtual reference paths, providing insights into path optimization performance. Shorter paths generally indicate more efficient task execution, though this must be balanced against safety considerations.

$$PL = \sum_{i=1}^{L-1} \|\mathbf{p}_{i+1} - \mathbf{p}_i\|_2$$

#### Path Smoothness
Path Smoothness[2] evaluates the rate of change in trajectory direction, detecting oscillations that may arise from velocity changes or directional adjustments[5]. The metric is calculated as the absolute distance between displacement vectors in subsequent trajectory points, normalized by total path length. Smooth trajectories are essential for safe robot operation and reduced mechanical wear.

$$PS = \frac{1}{PL}\sum_{i=1}^{L-2} \|(\mathbf{p}_{i+2} - \mathbf{p}_{i+1}) - (\mathbf{p}_{i+1} - \mathbf{p}_i)\|_2$$

#### Curvature Change
Curvature Change[6] provides specialized evaluation for mobile robots, measuring trajectory smoothness while accounting for robot orientation. This metric proves particularly valuable for car-like mobile robots where curvature directly relates to turning radius constraints. Unlike path smoothness, curvature change incorporates angular velocity, providing more comprehensive trajectory assessment.

$$CC = \frac{1}{L-2}\sum_{i=1}^{L-2}|\kappa_{i+1} - \kappa_i|, \quad \kappa_i = \frac{\theta_{i+1} - \theta_i}{\|\mathbf{p}_{i+1} - \mathbf{p}_i\|_2}$$

#### Trajectory Errors
Trajectory Error encompasses Absolute Trajectory Error (ATE) and Relative Trajectory Error (RTE), measuring deviations between predicted and reference trajectories[7]. ATE evaluates global consistency while RTE assesses local accuracy, both critical for navigation and manipulation tasks requiring precise positioning[8].

$$ATE = \frac{1}{L}\sum_{i=1}^{L} \|\mathbf{p}_i - \mathbf{p}_i^*\|_2$$

$$RTE = \frac{1}{L-\Delta}\sum_{i=1}^{L-\Delta} \|(\mathbf{p}_{i+\Delta} - \mathbf{p}_i) - (\mathbf{p}_{i+\Delta}^* - \mathbf{p}_i^*)\|_2$$

### 1.2.3 Vision-Language Alignment Metrics

#### BLEU Score
BLEU Score measures n-gram overlap between generated and reference captions, providing quantitative assessment of text generation quality[9]. BLEU-4 specifically evaluates 4-gram overlap, commonly used in image captioning tasks where robotics policies must describe observed scenes.

$$BLEU_n = BP \cdot \exp\left(\frac{1}{n}\sum_{k=1}^{n} \ln p_k\right)$$

#### CIDEr Score
CIDEr Score employs consensus-based evaluation, weighting n-grams using TF-IDF to prioritize informative content[10]. This metric proves particularly effective for evaluating descriptive capabilities of robotics policies in explaining their reasoning and observations.

$$CIDEr = \frac{1}{M}\sum_{m=1}^{M} \frac{\sum_g w_g^{(c)} w_g^{(m)}}{\|\mathbf{w}^{(c)}\|_2 \|\mathbf{w}^{(m)}\|_2}$$

#### METEOR
METEOR assesses semantic similarity between generated and reference text, accounting for synonyms and paraphrasing[11]. This metric provides more nuanced evaluation than simple n-gram matching, particularly valuable for natural language interaction capabilities.

$$METEOR = F_{\text{mean}}(1 - \gamma P_{\text{penalty}})$$

#### IoU
Intersection over Union measures object detection accuracy by evaluating overlap between predicted and ground truth bounding boxes[12]. This metric is fundamental for assessing visual perception capabilities that underpin robotics policy performance.

$$IoU = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$$

#### CLIP Score
CLIPScore utilizes pre-trained vision-language models to compute similarity between images and text in a shared embedding space. This metric effectively evaluates cross-modal alignment, crucial for robotics policies that must bridge visual observations and linguistic instructions[13].

$$CLIPScore = \cos(\mathbf{e}_{\text{img}}, \mathbf{e}_{\text{text}})$$

### 1.2.4 Safety and Robustness Metrics

#### Collision Rate
Collision Rate quantifies the frequency of collisions during task execution, serving as a primary safety indicator[14]. This metric is particularly critical for mobile robots and humanoids operating in human environments where safety is paramount.

$$CR = \frac{N_{\text{collisions}}}{T_{\text{steps}}}$$

#### Obstacle Proximity
Obstacle Proximity measures the minimum distance between the robot and environmental obstacles throughout task execution. This metric provides insights into safety margins and risk assessment capabilities of robotics policies in cluttered environments[15].

$$OP = \min_t d_t^{\text{robot→obstacle}}$$

#### Risk Factor
Risk Factor offers comprehensive safety evaluation[16] by integrating proximity measurements throughout the route. Calculated as the average of the reciprocal distances from obstacles, this metric provides a holistic assessment of safety-conscious behavior.

$$RF = \frac{1}{T}\sum_{t=1}^{T} \frac{1}{d_t}$$

### 1.2.5 Efficiency Metrics

#### Inference Latency
Inference Latency measures the time required to generate actions from visual observations and language instructions. This metric is crucial for real-time applications where responsive behavior is essential for effective human-robot interaction[?].

$$IL = t_{\text{infer,end}} - t_{\text{infer,start}}$$

#### Computation Time
Computation Time evaluates processing requirements per decision cycle[17], informing deployment considerations for resource-constrained environments. This metric helps determine feasibility for edge computing applications and consumer hardware deployment.

$$CT = \frac{\text{CPU/GPU cycles}}{\text{decision step}}$$

#### Memory Usage
Memory Usage assesses resource consumption during operation[18], particularly relevant for compact robotics policies like Smolrobotics-450M designed for consumer hardware deployment. Efficient memory usage enables broader accessibility and real-world deployment scenarios.

$$MU = \max_t(\text{RAM}_t + \text{VRAM}_t)$$

## References

[1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., "Rt-1: Robotics transformer for real-world control at scale," arXiv preprint arXiv:2212.06817, 2022.

[2] M. Dobiš, M. Dekan, P. Beňo, F. Duchoň, and A. Babinec, "Evaluation criteria for trajectories of robotic arms," Robotics, vol. 11, p. 29, Feb. 2022.

[3] K. K. A. Farag, H. H. Shehata, and H. M. El-Batsh, "Mobile robot obstacle avoidance based on neural network with a standardization technique," J. Robot., vol. 2021, pp. 1–14, Nov. 2021.

[4] P. Fankhauser, M. Bloesch, D. Rodriguez, R. Kaestner, M. Hutter, and R. Siegwart, "Kinect v2 for mobile robot navigation: Evaluation and modeling," in 2015 International Conference on Advanced Robotics (ICAR), IEEE, July 2015.

[5] S. Guillén Ruiz, L. V. Calderita, A. Hidalgo-Paniagua, and J. P. Bandera Rubio, "Measuring smoothness as a factor for efficient and socially accepted robot motion," Sensors (Basel), vol. 20, p. 6822, Nov. 2020.

[6] J.-H. Hwang, R. C. Arkin, and D.-S. Kwon, "Mobile robots at your fingertip: Bezier curve on-line trajectory generation for supervisory control," in Proceedings 2003 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2003) (Cat. No.03CH37453), IEEE, 2004.

[7] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, "A benchmark for the evaluation of RGB-D SLAM systems," in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, IEEE, Oct. 2012.

[8] F. Endres, J. Hess, N. Engelhard, J. Sturm, D. Cremers, and W. Burgard, "An evaluation of the RGB-D SLAM system," in 2012 IEEE International Conference on Robotics and Automation, IEEE, May 2012.

[9] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "Bleu: a method for automatic evaluation of machine translation," in Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, ACL '02, (USA), p. 311–318, Association for Computational Linguistics, 2002.

[10] R. Vedantam, C. Lawrence Zitnick, and D. Parikh, "Cider: Consensus-based image description evaluation," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2015.

[11] S. Banerjee and A. Lavie, "METEOR: An automatic metric for MT evaluation with improved correlation with human judgments," in Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pp. 65–72, Association for Computational Linguistics, June 2005.

[12] M. Berman, A. R. Triki, and M. B. Blaschko, "The lovász-softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[13] J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, and Y. Choi, "CLIPScore: A reference-free evaluation metric for image captioning," in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7514–7528, Association for Computational Linguistics, Nov. 2021.

[14] M. Hoy, A. S. Matveev, and A. V. Savkin, "Algorithms for collision-free navigation of mobile robots in complex cluttered environments: a survey," Robotica, vol. 33, pp. 463–497, Mar. 2015.

[15] N. Blunder, M. Thiel, M. Schrick, J. Hinckeldeyn, and J. Kreutzfeldt, "Integration and evaluation of a close proximity obstacle detection for mobile robots in public space," 2022.

[16] A. Majumdar and M. Pavone, "How should a robot assess risk? towards an axiomatic theory of risk in robotics," in Springer Proceedings in Advanced Robotics, pp. 75–84, Cham: Springer International Publishing, 2020.

[17] J. Hartmanis and R. E. Stearns, "On the computational complexity of algorithms," Trans. Am. Math. Soc., vol. 117, p. 285, May 1965.

[18] X.-H. Sun and D. Wang, "APC," Perform. Eval. Rev., vol. 40, pp. 125–130, Oct. 2012.
