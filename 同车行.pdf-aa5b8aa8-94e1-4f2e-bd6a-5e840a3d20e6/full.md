DSCC2017-5209

# A FINITE STATE MACHINE BASED AUTOMATED DRIVING CONTROLLER AND ITSSTOCHASTIC OPTIMIZATION

Mengxuan Zhang Department of Aerospace Engineering University of Michigan Ann Arbor, Michigan 48109 Email: mengxuan@umich.edu

Anouck Girard Department of Aerospace Engineering University of Michigan Ann Arbor, Michigan 48109 Email: anouck@umich.edu

Nan Li<sup>∗</sup> Department of Aerospace Engineering University of Michigan Ann Arbor, Michigan 48109 Email: nanli@umich.edu

Ilya Kolmanovsky Department of Aerospace Engineering University of Michigan Ann Arbor, Michigan 48109 Email: ilya@umich.edu

## ABSTRACT

In this paper, we develop a finite state machine based automated highway driving controller. The controller is described by feedback control laws in each state and state transition conditions. We test the controller in a traffic simulator and evaluate its performance based on a metric function. Furthermore, we propose a stochastic gradient based optimization approach to achieve optimal calibration of the developed controller. We expect that this controller can serve as a baseline for automated driving algorithm developments.

## 1 INTRODUCTION

Automated driving technologies have the potential to greatly improve the safety and comfort, as well as efficiency and fuel economy of everyday transportation. Various efforts have been put into the development of automated vehicle decision and control algorithms.

Some approaches for automated highway driving that have been explored in the literature are: 1) Markov decision process based decision making, as in [1–3]; 2) game theory based decision making, as in [4, 5]; 3) tree search based action selection, as in [6]; and, 4) rule-based control, as in [7].

Among these approaches, the rule-based controller, sometimes referred to as “finite state machine (FSM)” has its own advantages, including: 1) Clear in structure: the controller is based on “if-then-else” logic, which is explicitly readable, so that the controller’s behavior can be relatively easily predicted; 2) Easy to calibrate: a FSM usually has a finite number of parameters, so that it is easy to calibrate and optimize; 3) A well-calibrated FSM is usually more reliable, compared to some other frameworks, e.g., based on function approximation techniques as used in machine learning-based approaches. Note that this is to some extent also a result of 1) and 2). For further discussions on rulebased controllers, the reader is referred to [8].

In this work, we propose a FSM-based controller for automated driving on multi-lane highways. This controller is sufficiently concise in structure that it is easily implementable, and sufficiently robust to be able to drive the vehicle in an uncertain traffic environment safely with acceptable performance. It may serve as a baseline or a benchmark for automated driving algorithms, that is, a well-developed automated driving algorithm should be at least comparable in performance to this FSM controller.

In [7], a multi-agent traffic simulation system, called “Simulation of Intelligent TRAnsport Systems (SITRAS),” was developed, where the drivers’ behavior was modeled based on logical rules. Although the driving controller proposed in this paper and the driver model developed in [7] are both based on rules, the foci are different. In [7], the focus was on microscopic traffic simulation; while in this paper, our focus is to develop an automated driving controller. As a result, the car dynamics considered in this paper are at a more detailed level. For example, in [7], a lane change maneuver was assumed to be executed as an instantaneous change; while in this paper, we consider the dynamics during a lane change; in particular, we also consider decision making during lane changes.

The configuration, $\mathrm { e . g . }$ , the parameters, of an automated driving controller can be optimized using probabilistic simulation outcomes. In particular, the expected value of a metric function that measures the controller’s performance can be obtained for each configuration point through multiple simulation runs, and then the best point can be selected, as done in [9]. However, such an approach requires meshing the parameter space to obtain a finite number of configuration points, and may require a large amount of simulation time to obtain the expected metric function value for every configuration point and then compare them. In this paper, we propose a stochastic gradient based method to optimize the developed FSM-based controller, which can be more efficient to achieve its optimal calibration. For discussions about stochastic gradient optimization, see [10].

This paper is organized as follows: in Section 2, we introduce the FSM-based automated driving controller; in Section 3, we test the controller in a traffic simulator and show the simulation results; in Section 4, we use a stochastic gradient based method to optimize the controller; conclusions are given in Section 5.

## 2 AUTOMATED DRIVING CONTROLLER DESIGN

The problem we treat in this work is to develop a finite state machine based controller for an automated vehicle driving on a multi-lane highway. The objective is to provide a concisely implementable solution for automated highway driving, and may serve as a baseline or a benchmark for automated driving algorithms. In this section, we present the controller. In Section 3, we show the simulation results. In Section 4, we propose a method to optimize the controller based on a stochastic gradient optimization approach.

## 2.1 Car dynamics

In this work, we consider the following discrete-time equations of motion to represent the dynamics of a car,

$$
\begin{array}{c} x (t + 1) = x (t) + v _ {x} (t) \Delta t, \\ v _ {x} (t + 1) = v _ {x} (t) + a (t) \Delta t, \\ y (t + 1) = y (t) + v _ {y} (t) \Delta t, \end{array}\tag{1}
$$

where x and $\nu _ { x }$ represent the position and velocity of the car on the highway in the longitudinal direction, and y represents the position of the car in the lateral direction. The longitudinal acceleration, $a ( t )$ , and the lateral velocity, $\nu _ { y } ( t )$ , are control inputs that are determined by the driving controller.

## 2.2 Controller inputs

An automated vehicle driving on a multi-lane highway is able to perceive its surrounding environment through sensors it is equipped with, such as radars, lidars and cameras.

In this work, we propose the following sensing system based on indicators, inspired by [11]:

• The front center range and its time rate of change, called “range rate,” denoted by $d _ { \mathrm { f c } }$ and $\nu _ { \mathrm { f c } }$ . The front center range is the longitudinal distance from the ego vehicle to the vehicle immediately and directly in its front, called “front center vehicle.”

• The front left range and range rate, $d _ { \mathrm { f l } }$ and $\nu _ { \mathrm { f l } }$ . The front left range is the longitudinal distance from the ego vehicle to the vehicle in its front left, called “front left vehicle.”

• The front right range and range rate, $d _ { \mathrm { f r } }$ and $\nu _ { \mathrm { f r } }$ . The front right range is the longitudinal distance from the ego vehicle to the vehicle in its front right, called “front right vehicle.”

• The rear left range (longitudinal distance to the vehicle in the rear left, called “rear left vehicle”), $d _ { \mathrm { r l } }$

• The rear right range (longitudinal distance to the vehicle in the rear right, called “rear right vehicle”), $d _ { \mathrm { r r } }$

• The distance to left constraints, $d _ { \mathrm { l } } .$

• The distance to right constraints, $d _ { \mathrm { r } } .$ .

• The lateral position of the ego vehicle on the road, y.

• The longitudinal velocity of the ego vehicle, $\nu _ { x } .$

Fig. 1 to Fig. 3 show examples of traffic scenarios on a highway with three lanes. The red car in the center represents the ego vehicle, and the yellow cars represent the traffic environment.

Fig. 1 shows a scenario where every car in the traffic travels in the center of a lane. The front ranges are measured as the bumper-to-bumper distances from the ego vehicle to vehicles in front. Similarly, the rear ranges are measured as the bumper-tobumper distances from the ego vehicle to vehicles in its back. When no vehicle can be detected at some position because of limited sensor range, e.g., the front right position in Fig. 1, the corresponding range is set to be the maximum visual range of the sensor, $d _ { \mathrm { s i g h t } }$ ; meanwhile, the corresponding range rate is set to be 0.

![](images/5962fd3242d1c0870f453d965ac012e0cea77283f160f5a4c2b09a61e9d37efe.jpg)  
FIGURE 1. Traffic scenario example where every car in the traffic travels in the center of a lane.

![](images/54fe0d24b6a3012381673d303799c336332e9fa569db74e0ad49a5df66c117ea.jpg)  
FIGURE 2. Traffic scenario example where the “front center vehicle” is the same car as the “front right vehicle.”

The distance to left/right constraints is defined as follows: as shown in Fig. 1, when there is no car in a parallel position to the left/right of the ego vehicle, the corresponding distance to constraints is set to be the distance from the edge of the ego vehicle to the road boundary; while when there is a car in a parallel position to the left/right of the ego vehicle, the corresponding distance to constraints is set to be the distance from the edge of the ego vehicle to that car’s edge.

We remark that the observation indicators defined in some references appear to only be able to handle the above scenarios, i.e., every car (including the ego vehicle and other cars in traffic) drives in the center of a lane. When the ego vehicle is experiencing a lane change, or when some other cars in the vicinity of the ego car are experiencing lane changes, there may be some ambiguity in the definition of some observation indicators, for example: in the scenarios shown in Fig. 2 and Fig. 3, which car is treated as the “front left/center/right vehicle” needs to be carefully defined.

In particular, we define the front sensor range to be the rectangular area directly in front of the ego vehicle as shown in Fig. 2, that is, its left/right boundary is collinear with the left/right edge of the ego vehicle; we define the front left sensor range to be the rectangular area as shown in Fig. 3, such that its right boundary is collinear with the ego vehicle’s geometric center and its left boundary is deviated from its right boundary by the width of a lane; the front right and the rear left/right sensor ranges are defined in a similar way.

![](images/1a6d3874dde397154b3dda4d31fc524b2b65c5152c8ccf2a5533d8bad0eaab1d.jpg)  
FIGURE 3. Traffic scenario example where the “front left vehicle” is the closest car in the front left sensor range.

For the controller developed in this paper, the front center vehicle is defined as the closest car in front of the ego vehicle whose geometric contour has overlap with the front sensor range of the ego vehicle. The front left/right vehicle is defined as the closest car in front of the ego vehicle whose geometric contour has overlap with the front left/right sensor range of the ego vehicle. The definitions for the rear left/right vehicles are similar.

Note that under these definitions, the front center vehicle and front left/right vehicle may actually be the same vehicle. Fig. 2 shows such a case where a yellow car in the front right position is cutting into the lane where the red ego vehicle travels. The cutting vehicle is considered as both the front center vehicle and the front right vehicle for defining the corresponding range and range rate input values.

Another example is shown in Fig. 3: the yellow car in the leftmost position is identified to be the front left vehicle of the red car. However, if it goes further left until it does not overlap with the red dashed line (that indicates the boundary of the front left sensor range of the ego vehicle), the “front left vehicle” will be switched to the yellow car in the “front” of the figure.

We remark that the controller inputs under the above definitions are without ambiguity no matter whether the cars in traffic (including the ego vehicle and other cars in its vicinity) are driving in the center of lanes or performing lane changes. This is a necessary requirement to let the automated vehicle be able to drive safely on the highway.

## 2.3 Finite state machine controller

The automated driving controller that we design in this paper is a finite state machine (FSM). Mathematically, the FSM defined in this paper is described by a tuple $( P , M , m _ { 0 } , \delta )$ , where P denotes the input set, M denotes the state set, $m _ { 0 }$ is the initial state, and $\delta : M \times P \to M$ represents the state transition function. In particular, we define each state (also referred to as “mode”) by a set of control laws to control the longitudinal and lateral motion of the vehicle in this state, see Section 2.3.1; the input set is a set of Boolean variables concerning the traffic environment, and the state transition function is represented by a set of mode switch conditions based on the truth values of the Boolean variables, see Section 2.3.2. The FSM diagram is shown in Fig. 4.

![](images/d6060e2e67451d86eeb63a587b7df4b0260939027e5b00ab075abf3691d27ead.jpg)  
FIGURE 4. Diagram for the finite state machine automated highway driving controller. Items [1-15] are mode switch conditions to be designed.

The control laws for each mode and the switch conditions between the modes are introduced in the following sections.

## 2.3.1 Finite state machine control modes

Adaptive cruise control mode – A In the adaptive cruise control mode (A), the ego vehicle follows a preceding car and keeps a desired time headway, $t _ { \mathrm { d e s } }$

The longitudinal acceleration and the lateral velocity of the ego vehicle are determined by Eqn. (2) and Eqn. (3),

$$
a (t) = K _ {\mathrm{p}} \left(d _ {\mathrm{fc}} (t) - t _ {\mathrm{des}} v _ {x} (t)\right) + K _ {\mathrm{v}} v _ {\mathrm{fc}} (t),\tag{2}
$$

$$
v _ {y} (t) = 0,\tag{3}
$$

where $K _ { \mathfrak { p } }$ is the gain to match the actual car-following distance, $d _ { \mathrm { f c } } ( t )$ , to a desired distance determined by the desired time headway, $d _ { \mathrm { d e s } } ( t ) = t _ { \mathrm { d e s } } \nu _ { x } ( t )$ , while $K _ { \mathrm { v } }$ is the gain to match the ego vehicle’s longitudinal velocity to the velocity of the preceding car.

In particular, the ac/deceleration rate of the vehicle is assumed to be bounded by $a \in [ - a _ { \mathrm { m a x } } , a _ { \mathrm { m a x } } ]$ . If the computed acceleration based on (2) is outside the bounds, it gets saturated. In addition, the longitudinal velocity of the vehicle is assumed to be bounded by $\nu _ { x } \in [ \nu _ { \operatorname* { m i n } } , \nu _ { \operatorname* { m a x } } ]$ , and gets saturated when the computed value is outside the bounds.

Cruise control mode – C When there is no car in front of the ego vehicle, the cruise control mode (C) is used to maintain a reference longitudinal speed, $\nu _ { \mathrm { r e f } } .$

The control laws in the cruise control mode are as follows,

$$
\begin{array}{c} a (t) = K _ {\mathrm{c}} \big (v _ {\mathrm{ref}} - v _ {x} (t) \big), \\ v _ {y} (t) = 0, \end{array}\tag{4}
$$

(5)

where $K _ { \mathrm { c } }$ is the gain to match the ego vehicle’s longitudinal velocity, $\nu _ { x } ( t )$ , to a reference velocity, $\nu _ { \mathrm { r e f } }$ . The acceleration and velocity are bounded in the same way as in the adaptive cruise control mode. In particular, in this work, we set $\nu _ { \mathrm { r e f } } = \nu _ { \mathrm { m a x } }$

Lane change modes – L and R The left/right lane change modes (L and $R )$ are defined to control the ego vehicle’s motion when it performs lane changes. When the ego vehicle changes lanes from the original lane to a target lane – either the lane on its left or the lane on its right – its motion is controlled as follows:

$$
a (t) = K _ {\mathrm{p}} \left(d _ {\mathrm{fc}} (t) - t _ {\mathrm{des}} v _ {x} (t)\right) + K _ {\mathrm{v}} v _ {\mathrm{fc}} (t),
$$

$$
v _ {y} (t) = \pm \frac {w}{t _ {\mathrm{cl}}},\tag{6}
$$

(7)

where w is the width of a lane, and $t _ { \mathrm { c l } }$ is the time needed to make a lane change. The plus/minus sign indicates the lane change direction. In particular, the longitudinal control used, Eqn. (6), is the same as that of the adaptive cruise control mode, Eqn. (2), as well as the saturation conditions.

Lane change pause mode – P An intended lane change may be interrupted due to various reasons. For example, the rear car in the target lane starts to overtake the ego vehicle, in which case the ego vehicle’s lane change would cut off the overtaking; or, a car in front starts to move into the target lane, in which case the ego vehicle would be blocked by that car after its lane change. In these situations, the lane change maneuver should be paused, that is, to set the lateral velocity back to zero.

After that, the ego vehicle makes new lane change decisions, responding to these situations. The ego vehicle either waits for another chance to continue its unfinished lane change, or abort it, depending on the situation, until it successfully moves to the center of a lane. In the lane change pause mode, the longitudinal motion is controlled in the same way as in the adaptive cruise control mode, while the lateral velocity is set to zero,

$$
a (t) = K _ {\mathrm{p}} \left(d _ {\mathrm{fc}} (t) - t _ {\mathrm{des}} v _ {x} (t)\right) + K _ {\mathrm{v}} v _ {\mathrm{fc}} (t),\tag{8}
$$

$$
v _ {y} (t) = 0.\tag{9}
$$

## 2.3.2 Mode switch conditions

In this subsection, we introduce the thresholds for switching between modes. The controller’s mode at time t is denoted by $M ( t )$ . The $p _ { i } , i = 1 , \ldots , 6 ,$ , are Boolean variables and each represents the truth value (0 or 1) of a specific condition. In particular, the following variables are used to determine the controller’s mode and mode transitions;

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
$p_{1}: d_{\mathrm{fc}} &lt; t_{\mathrm{cc}}v_{x} - d_{\mathrm{over}},$ $p_{2}: d_{\mathrm{fc}} &gt; t_{\mathrm{cc}}v_{x} + d_{\mathrm{over}},$ $p_{3\mathrm{l}}: a_{\mathrm{l}} &gt; a + a_{\mathrm{over}},$ $p_{3\mathrm{r}}: a_{\mathrm{r}} &gt; a + a_{\mathrm{over}}, \text { and } a_{\mathrm{r}} &gt; a_{\mathrm{l}},$ $p_{4\mathrm{l}}: d_{\mathrm{fl}} &gt; t_{\mathrm{win}}v_{x} \text { and } d_{\mathrm{rl}} &gt; t_{\mathrm{win}}v_{x},$ $p_{4\mathrm{c}}: d_{\mathrm{fc}} &gt; t_{\mathrm{win}}v_{x},$ $p_{4\mathrm{r}}: d_{\mathrm{fr}} &gt; t_{\mathrm{win}}v_{x} \text { and } d_{\mathrm{rr}} &gt; t_{\mathrm{win}}v_{x},$ $p_{5}: y \in \text { at the center of a lane within a deviation tolerance, }$ $p_{6\mathrm{l}}: d_{\mathrm{l}} &gt; d_{\mathrm{side}},$ $p_{6\mathrm{r}}: d_{\mathrm{r}} &gt; d_{\mathrm{side}},$
</div>

where $t _ { \mathrm { c c } }$ is a critical time headway such that when the actual time headway $\frac { d _ { \mathrm { f c } } } { \nu _ { x } }$ is less than $t _ { \mathrm { c c } }$ , the ego vehicle switches from the cruise control mode (C) to the adaptive cruise control mode (A) to perform car following; on the other hand, if the actual time headway $\frac { d _ { \mathrm { f c } } } { \nu _ { x } }$ is larger than $t _ { \mathrm { c c } } .$ , the ego vehicle switches from the adaptive cruise control mode to the cruise control mode to maintain a reference speed. The $d _ { \mathrm { o v e r } }$ denotes an overshoot value between switches, to create a dead band that avoids back-andforth jumps between the A and C modes.

The a<sub>l</sub> (a<sub>r</sub>) is the predicted acceleration that the ego vehicle can achieve if it makes a lane change to the left (right), supposing the left (right) lane exists. The $a _ { \mathrm { l } } \ ( a _ { \mathrm { r } } )$ is defined in Eqn. (10) (Eqn. (11)), if $d _ { \mathrm { f l } } ( t ) < t _ { \mathrm { c c } } \nu _ { x } ( t ) ( d _ { \mathrm { f r } } ( t ) < t _ { \mathrm { c c } } \nu _ { x } ( t )$ ), and is bounded by $[ - a _ { \mathrm { m a x } } , a _ { \mathrm { m a x } } ]$ in the same way as defined in the previous section,

$$
a _ {\mathrm{l}} (t) = K _ {\mathrm{p}} \left(d _ {\mathrm{fl}} (t) - t _ {\mathrm{des}} v _ {x} (t)\right) + K _ {\mathrm{v}} v _ {\mathrm{fl}} (t),\tag{10}
$$

$$
a _ {\mathrm{r}} (t) = K _ {\mathrm{p}} \left(d _ {\mathrm{fr}} (t) - t _ {\mathrm{des}} v _ {x} (t)\right) + K _ {\mathrm{v}} v _ {\mathrm{fr}} (t).\tag{11}
$$

On the other hand, if a lane change to the left/right lane lets the ego vehicle drive in the cruise control mode (C), i.e., there is no car in that lane in front that blocks the travel of the ego vehicle, we set $a _ { \mathrm { l } } = a _ { \mathrm { m a x } } \ : \mathrm { o r } \ : a _ { \mathrm { r } } = a _ { \mathrm { m a x } }$ , correspondingly. The motivation for performing a lane change is the potential to achieve a higher travel speed. Thus, we compare $a ( t )$ to $a _ { \mathrm { l } } ( t )$ and $a _ { \mathrm { r } } ( t )$ to decide whether to make lane changes. In particular, the constant $a _ { \mathrm { o v e r } }$ is a threshold such that only when the speed benefit is sufficiently large does the ego vehicle perform a lane change; otherwise, the ego vehicle stays in its current lane to reduce maneuvers and improve comfort.

Before making a lane change, the ego vehicle should consider the safety of performing such a lane change. The $t _ { \mathrm { w i n } }$ represents a minimum time window required to safely complete a lane change. Similarly, $d _ { \mathrm { s i d e } }$ represents a safety margin of distance to side constraints to further improve the lane change safety.

The switch conditions between modes are summarized in Table 1, where each column corresponds to the current mode $M ( t )$ , and each row corresponds to the mode at the next time step, M(t + 1). The entries with “NA” indicate that the corresponding mode switch is “not available,” and “OW” means “otherwise.”

We also remark that when left and right lane changes are both possible, the ego vehicle prefers lane change to its left when overtaking, and prefers lane change to its right when aborting the previous lane change from the lane change pause mode.

## 3 RESULTS

In this section, we present the simulation results of using the proposed FSM controller to drive an automated vehicle on a three-lane highway. We assume that the traffic is constituted by multiple other cars driving on the same highway and driving in the same direction.

## 3.1 Simulation environment

To test the performance of the developed automated driving controller, we exploit a traffic simulator developed in [9, 12]. In this subsection, we briefly introduce the simulator and its functionality.

The traffic simulator is constructed based on a game theoretic traffic model, where the drivers are modeled as interactive strategic decision makers. In particular, hierarchical reasoning game theory, also referred to as “level-k” game theory, is exploited to obtain the driver models.

TABLE 1. Mode switch conditions

<table><tr><td></td><td>A</td><td>C</td><td>L</td><td>R</td><td>P</td></tr><tr><td>A</td><td>OW</td><td> $p_{1}$ </td><td> $p_{5}$ </td><td> $p_{5}$ </td><td>NA</td></tr><tr><td>C</td><td> $p_{2}$ </td><td>OW</td><td>NA</td><td>NA</td><td>NA</td></tr><tr><td>L</td><td> $(\neg p_{2})p_{3\text{l}}p_{4\text{l}}p_{4\text{c}}p_{6\text{l}}$ </td><td>NA</td><td>OW</td><td>NA</td><td> $p_{6\text{l}} \neg p_{6\text{r}}$ </td></tr><tr><td>R</td><td> $(\neg p_{2})p_{3\text{r}}p_{4\text{r}}p_{4\text{c}}p_{6\text{r}}$ </td><td>NA</td><td>NA</td><td>OW</td><td> $p_{6\text{r}}$ </td></tr><tr><td>P</td><td>NA</td><td>NA</td><td> $\neg p_{5} \neg p_{6\text{l}}$ </td><td> $\neg p_{5} \neg p_{6\text{r}}$ </td><td>OW</td></tr></table>

A driver of level-k reacts to its surrounding environment by assuming all of the other drivers are level-(k-1) and making decisions as the best response to their potential actions. This way, every driver in the traffic interacts with each other as well as with the test car controlled by the developed FSM controller. Compared to some other traffic models in the literature, such a traffic model is expected to have higher fidelity (by modeling the decision making processes of human drivers and driver-to-driver interactions) and to be able to create a pool of various traffic scenarios with closed-loop traffic state transitions (because the traffic model also reacts to the test car’s actions).

In [9,12], such a traffic model is used to build up a simulator to test and evaluate automated driving algorithms. In particular, two algorithms, based on Stackelberg game theory and decision trees, have been tested and compared. In this work, we also exploit this simulator to test the developed FSM automated driving controller presented in the previous sections. In [9], it was shown that the parameters of a given control algorithm can be optimized based on a reward function and the probabilistic simulation outcomes. In this paper, we propose a more efficient way to optimize the controller.

In particular, we set up the environment as a three-lane highway. The car controlled by the proposed FSM controller is the ego vehicle, while all other cars on the road form the traffic environment. The width of a lane, w, is 3.6 m, and all cars are modeled as 6 × 2 m rectangles. The travel speed for each car is bounded by [62, 98] km/h. The cars in the traffic always drive in the center of a lane unless they are changing lanes. Each lane change is assumed to take 2 seconds with constant lateral velocity.

The parameters for the control laws in different modes are summarized in Table 2, and the parameters for mode switches are summarized in Table 3.

## 3.2 Simulation results

We run each simulation for 200 seconds with $\Delta t = 1 . 0$ sec. Fig. 5 shows the time histories of the control modes of two simulation runs of different traffic densities. Fig. 5(a) corresponds to sparse traffic, which is composed of 15 cars apart from the ego vehicle. We can see that the ego vehicle stays in the cruise control mode (C) at the beginning of the simulation. $\mathbf { A } \mathfrak { t } ~ t = 4 3$ sec, a car in front that travels with a lower speed blocks the ego vehicle and triggers the condition $p _ { 1 }$ , so the FSM switches to the adaptive cruise control mode (A) for one step. After that, the ego vehicle decides to make a lane change to overtake the slower car, and returns back to the C mode after completing the lane change.

TABLE 2. Finite state machine control parameters

<table><tr><td>Parameter</td><td>Value</td><td>Units</td></tr><tr><td> $K_c$ </td><td>0.3</td><td> $sec^{-1}$ </td></tr><tr><td> $K_p$ </td><td>0.1</td><td> $sec^{-2}$ </td></tr><tr><td> $K_v$ </td><td>0.1</td><td> $sec^{-1}$ </td></tr><tr><td> $t_{des}$ </td><td>2</td><td>sec</td></tr><tr><td> $t_{cl}$ </td><td>2</td><td>sec</td></tr></table>

TABLE 3. Finite state machine mode switch parameters

<table><tr><td>Parameter</td><td>Value</td><td>Units</td></tr><tr><td> $t_{cc}$ </td><td>3.5</td><td>sec</td></tr><tr><td> $d_{over}$ </td><td>3</td><td>m</td></tr><tr><td> $a_{over}$ </td><td>2</td><td>m/sec $^{2}$ </td></tr><tr><td> $t_{win}$ </td><td>1</td><td>sec</td></tr><tr><td> $d_{side}$ </td><td>1.8</td><td>m</td></tr><tr><td> $a_{max}$ </td><td>4</td><td>m/sec $^{2}$ </td></tr></table>

At around 60 sec, the ego vehicle makes two left lane changes consecutively. We plot the snapshots of the simulation from 59 sec to 64 sec in Fig. 8. The red car represents the ego vehicle that is controlled by the FSM controller, while all the yellow cars are the traffic environment. In the snapshots, all of the cars travel to the right. We fix the frame on the ego vehicle. The motions of the other cars in traffic can be tracked from their relative motions to the ego vehicle. We can see that after the ego vehicle makes the first left lane change, one car in front and in the leftmost lane also makes a lane change to the middle lane and blocks the ego vehicle. Therefore, the ego vehicle makes another left lane change to overtake it.

Fig. 5(b) shows the time history of the control modes of another simulation run where the traffic is dense (composed of 30 cars apart from the ego vehicle). We can see that the ego vehicle stays in the adaptive cruise control mode (A) for most of the time. The ego vehicle rarely makes lane changes in this simulation because the speed benefit from lane changes is low, and the time window needed to complete a lane change is rarely available.

![](images/f9a8a459945e377498f401d2bfbfa7921a6cb0408bd9f4163f5c2e351af96619.jpg)  
(a)

(b)  
![](images/2961460daa1d7d58bfc642b5707a625dd7f16f5988e482e51e57415c09537d90.jpg)  
FIGURE 5. The time history of the control mode of two simulation runs with different traffic.

## 3.3 Performance evaluation metric

We define an evaluation function as a metric to measure the controller’s performance. In particular, the evaluation function is a linear combination of two terms: 1) average travel distance before a safety violation, $R _ { \mathrm { d i s t } }$ , and 2) total acceleration and deceleration effort, $R _ { \mathrm { a c c } }$ . These two terms are defined by

$$
R _ {\text { dist }} = \frac {x _ {\text { total }}}{c _ {\text { total }} + 1},\tag{12}
$$

where $x _ { \mathrm { { t o t a l } } }$ is the total longitudinal distance traveled, $c _ { \mathrm { t o t a l } }$ is the total number of safety violations (the ego vehicle collides with some other car), and

$$
R _ {\mathrm{acc}} = - \sum_ {t = 0} ^ {T} a ^ {2} (t),\tag{13}
$$

where T is the total number of simulation steps, and $a ( t )$ is the acceleration at step t.

We remark that the first term, $R _ { \mathrm { d i s t } } .$ , reflects the controller’s performance in terms of both safety and travel speed, while the second term reflects the controller’s performance for maneuver effort, passenger comfort, and fuel efficiency. The evaluation function is a weighted sum of these two factors,

$$
R = w _ {1} R _ {\mathrm{dist}} + w _ {2} R _ {\mathrm{acc}},\tag{14}
$$

where $w _ { 1 }$ and w<sub>2</sub> are weighting factors. In the following results, we set $w _ { 1 } = 1$ and $w _ { 2 } = 0 . 1$

We test the developed FSM controller using traffic models of different aggressiveness levels, and evaluate its performance using the evaluation function (14).

We remark that the aggressiveness of the game theoretic traffic model in [9,12] can be varied by using driver models of different level with different proportions. In general, a level-1 driver is most aggressive; a level-0 driver is least aggressive (mild); and a level-2 driver is in between. To be “aggressive” means that the driver tends to make more lane changes and accelerations to overtake others.

In the following simulations, the “aggressive” traffic is composed of 66.6% level-1 drivers, and 33.3% level-2 drivers; the “mild” traffic is composed of 33.3% level-0 drivers, 33.3% level-1 drivers, and 33.3% level-2 drivers. Each simulation is 200 second long with time step $\Delta t = 1 . 0$ sec.

The relationship between the number of cars in the traffic (reflecting traffic density) and the evaluation function value (we call it the “reward” value in the plot, since the higher it is the better) is plotted in Fig. 6. Each point is obtained after 10,000 simulation runs to compute the average. We can see that 1) the value decreases as the traffic becomes denser; 2) the traffic aggressiveness does not have much influence on the values. The reason for this may be that the developed controller is based on strict logical rules and its performance is not sensitive to how the other cars behave in the traffic environment. Moreover, we observe that when the traffic is more aggressive, the ego vehicle achieves slightly higher reward values. One explanation for this may be that when the other cars in the traffic drive more aggressively, the overall traffic speed is increased, and thus the ego vehicle can also drive faster and obtains higher reward values as a result.

We also record the time percentage of each mode during the simulations, as presented in Table 4. We can see that 1) in over 85% of the time, the ego vehicle is controlled in the adaptive cruise control mode; 2) the time percentages of each mode are close in sparse traffic and in dense traffic, which indicates that the developed FSM controller has good robustness against traffic uncertainties; 3) the lane change pause mode is needed, to keep the ego vehicle safe.

We remark that the introduction of the lane change pause mode is motivated by the problematic traffic scenario for automated driving algorithms uncovered in [9]: the ego vehicle travels originally in the rightmost lane and decides to make a left lane change to the middle lane, while at the same time, another car originally in the leftmost lane also decides to change lane to the middle. In such a situation, the lane change pause mode could pause the ego vehicle’s lane change to avoid the danger of a side collision.

![](images/a458ef4cc6987f8e32f150f8a2ff8ff5c375095f085117f845de088bbe3338a0.jpg)  
FIGURE 6. The evaluation function values versus the number of cars in traffic.

TABLE 4. Control mode time percentage in different traffic environment.

<table><tr><td>Mode</td><td>15/Aggre.</td><td>15/Mild</td><td>30/Aggre.</td><td>30/Mild</td></tr><tr><td>C</td><td>3.27%</td><td>1.69%</td><td>3.79%</td><td>1.67%</td></tr><tr><td>A</td><td>84.57%</td><td>90.06%</td><td>85.47%</td><td>90.89%</td></tr><tr><td>L</td><td>5.96%</td><td>4.01%</td><td>5.18%</td><td>3.45%</td></tr><tr><td>R</td><td>6.19%</td><td>4.22%</td><td>5.52%</td><td>3.96%</td></tr><tr><td>P</td><td>0.01%</td><td>0.01%</td><td>0.03%</td><td>0.03%</td></tr></table>

## 4 STOCHASTIC GRADIENT OPTIMIZATION

The developed FSM controller is described by the control gains in each mode and the thresholds for switching between modes. We can optimize the controller’s performance by optimizing the values of these parameters based on an objective function, e.g., the evaluation function of Eqn.(14).

In [9], surfaces of objective function values versus the parameter values to be optimized are plotted. One can find the best combination of the parameter values by picking the points on the surfaces with maximum objective function values. These surfaces are obtained as the mean values of probabilistic outcomes of multiple simulation runs.

We note that when the dimension of the vector to be optimized is large, such an approach is time consuming, because we need to evaluate every point of a high dimensional mesh. In this paper, we use a method based on stochastic gradient optimization techniques to optimize the parameter values of the developed controller. With this approach, a locally optimal configuration of the controller can be obtained.

Define $w \in \mathbb { R } ^ { s }$ to be the vector that collects all the parameters to be optimized, e.g., feedback gains, $K _ { \mathrm { c } } , \ K _ { \mathrm { p } }$ or $K _ { \mathrm { v } } ,$ , or mode switch thresholds, such as $t _ { \mathrm { c c } } , ~ a _ { \mathrm { o v e r } } ,$ etc. Here, s indicates the number of parameters to optimize.

The objective function value R depends on the value of w. In particular, the objective function value is the mean value of probabilistic outcomes of multiple simulation runs, i.e.,

$$
R (w) = \frac {1}{n} \sum_ {i = 1} ^ {n} R _ {i} (w),\tag{15}
$$

where $R _ { i } ( w )$ is a random variable whose value depends on the probabilistic simulation outcomes.

A batch gradient based method to refine the value of w performs the following iterations,

$$
w ^ {k + 1} = w ^ {k} + \eta \nabla R (w) = w ^ {k} + \frac {\eta}{n} \nabla \left(\sum_ {i = 1} ^ {n} R _ {i} (w)\right),\tag{16}
$$

where the superscript k denotes the number of iterations, and $\eta >$ 0 is the update step size.

A stochastic gradient based method approximates the true gradient by a sample, such that

$$
w ^ {k + 1} = w ^ {k} + \eta \nabla R _ {i} (w).\tag{17}
$$

In this paper, we use a “mini-batch” gradient based method, such that the algorithm performs an update for every mini-batch of m samples, to achieve more stable convergence, that is,

$$
w ^ {k + 1} = w ^ {k} + \frac {\eta}{m} \nabla \left(\sum_ {i = 1} ^ {m} R _ {i} (w)\right).\tag{18}
$$

In this paper, a mini-batch is 1,000 simulation runs where each simulation run lasts 200 seconds. The direction of ascent is obtained by perturbing each component of $\cdot _ { w } k$ to both the positive direction and the negative direction a little bit, so that $2 ^ { s }$ sample points are evaluated and compared. The point with the highest objective function value indicates the ascent direction to update the value of w.

![](images/93daecdeeade6b6fcd276ef47413527291f6c68d3acbd9f55f115dc098f54f4f.jpg)  
FIGURE 7. The evolution of $\bar { R } ^ { k }$ during the optimization.

The stopping criterion of the algorithm is based on the convergence of the average objective function value over a past window, that is, the average objective function value of the past 100 iterations is recorded, denoted by $\bar { R } ^ { k }$ at iteration $k ,$ and the algorithm stops if the change in $\bar { R } ^ { k }$ between two iterations is less than a tolerance, i.e.,

$$
\left| \bar {R} ^ {k} - \bar {R} ^ {(k - 1)} \right| <   \varepsilon , \quad \text {   for   some   } \varepsilon > 0.\tag{19}
$$

In this paper, we pick three parameters to optimize – the gains in the adaptive cruise control mode, $K _ { \mathfrak { p } }$ and $K _ { \mathrm { v } } .$ , and the acceleration overshoot for lane change, $a _ { \mathrm { o v e r } } .$ . The traffic model used for optimization is composed of 15 cars, and the driver level percentage is the same as the “mild traffic” defined in the previous section.

Fig. 7 shows the evolution of $\bar { R } ^ { k }$ during the optimization. The vector $w = [ K _ { \mathrm { p } } , K _ { \mathrm { v } } , a _ { \mathrm { o v e r } } ]$ starts from [0.10, 0.20, 2.0] and converges to [0.061, 0.146, 2.62] after 190 iterations.

## 5 CONCLUSIONS

In this paper, we developed a finite state machine based controller for automated highway driving. Five modes were defined where each mode corresponds to particular control laws to perform cruise control, car following adaptive cruise control or lane changes; the switch conditions between modes were also defined.

We tested the controller in a game theoretic traffic simulator and the developed controller was shown to be robust against uncertainties in traffic. We evaluated the performance of the controller using an evaluation function that considers safety, performance, comfort and efficiency of the vehicle.

Finally, we proposed a stochastic gradient based approach to optimize the controller based on an objective function, to achieve a locally optimal calibration of the controller.

This FSM-based controller is concise in structure and relatively easy to implement. We expect that it can serve as a baseline or a benchmark for automated driving controller developments.

## REFERENCES

[1] S. Brechtel, T. Gindele, and R. Dillmann, “Probabilistic mdp-behavior planning for cars,” in Intelligent Transportation Systems (ITSC), 2011 14th International IEEE Conference on. IEEE, 2011, pp. 1537–1542.

[2] J. Wei, J. M. Dolan, J. M. Snider, and B. Litkouhi, “A point-based mdp for robust single-lane autonomous driving behavior under uncertainties,” in Robotics and Automation (ICRA), 2011 IEEE International Conference on. IEEE, 2011, pp. 2586–2592.

[3] S. Brechtel, T. Gindele, and R. Dillmann, “Probabilistic decision-making under uncertainty for autonomous driving using continuous pomdps,” in Intelligent Transportation Systems (ITSC), 2014 IEEE 17th International Conference on. IEEE, 2014, pp. 392–399.

[4] J. H. Yoo and R. Langari, “Stackelberg game based model of highway driving,” in Proc. ASME Dynamic Systems and Control Conference joint with JSME Motion and Vibration Conference, Fort Lauderdale, Florida, Oct. 2012.

[5] ——, “A stackelberg game theoretic driver model for merging,” in Proc. ASME Dynamic Systems and Control Conference, Palo Alto, California, Oct. 2013.

[6] L. Claussman, A. Carvalho, and G. Schildbach, “A path planner for autonomous driving on highway human mimicry approach with binary decision diagrams,” in Proceedings of the European Control Conference, Linz, Austria, July 2015.

[7] P. Hidas, “Modelling lane changing and merging in microscopic traffic simulation,” Transportation Research Part C: Emerging Technologies, vol. 10, no. 5, pp. 351–371, 2002.

[8] F. Hayes-Roth, “Rule-based systems,” Communications of the ACM, vol. 28, no. 9, pp. 921–932, 1985.

[9] N. Li, D. Oyler, M. Zhang, Y. Yildiz, I. Kolmanovsky, and A. Girard, “Game-theoretic modeling of driver and vehicle interactions for verification and validation of autonomous vehicle control systems,” arXiv preprint arXiv:1608.08589, 2016.

[10] J. C. Spall, Introduction to stochastic search and optimization: estimation, simulation, and control. John Wiley & Sons, 2005, vol. 65.

[11] C. Chen, A. Seff, A. Kornhauser, and J. Xiao, “Deep-

59[s]

driving: Learning affordance for direct perception in autonomous driving,” in Proceedings of the IEEE International Conference on Computer Vision, 2015, pp. 2722– 2730.

[12] N. Li, D. Oyler, M. Zhang, Y. Yildiz, A. Girard, and I. Kolmanovsky, “Hierarchical reasoning game theory based approach for evaluation and testing of autonomous vehicle control systems,” in Decision and Control (CDC), 2016 IEEE 55th Conference on. IEEE, 2016, pp. 727–733.

![](images/5966c0d570e156305fc48f159cfaab39105eb7e12efc22c09791d71ebce548bf.jpg)

![](images/f5f07bc7687df307f0f17f6734279fcfddd2db43908e23f67e35adec8cf6eb50.jpg)

![](images/055b5cdedadaaf788dc59be028d5e29d25d20facf614dc5cefc394cfad0e37de.jpg)

![](images/766ee5fbca8f2d2b8892f0d8cad3b121cdb206cfa112eafdc412cf3b269f6f3e.jpg)

![](images/cc85536e49ec8c41703a33d2570932f3269f4bc0c19bba039980fce5edcc43b4.jpg)

![](images/f2eb51234cd5d51cea8ac76ef43386a85abb3ca710813696588fa6cbb90ae8f3.jpg)  
FIGURE 8. Snapshots of the simulation.