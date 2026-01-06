# COViz-taxi: Demystifying AI Decisions

A non-technical guide to understanding how our AI Taxi driver learns and decides.

````carousel
# 1. The "AI Taxi Driver" Concept
Imagine a self-driving taxi in a simple grid world. 
- It wants to pick up a passenger and drop them off at a destination.
- It earns points (rewards) for doing it right.
- It loses points for crashing or taking too long.

**The Goal**: We want to know *why* the AI chose to go Left instead of Right. Was it smart? Or was it just luck?
<!-- slide -->
# 2. The Power of "What If?"
To understand the AI's choice, we don't just watch what it did. We ask: 
**"What if you had done the OTHER thing?"**

- **Actual Path**: The AI went North.
- **Counterfactual Path**: We force the AI to go South for one step, then let it drive normally.

By comparing these two parallel universes, we can see which path was actually better.
<!-- slide -->
# 3. Visualizing the Difference
Our program, **COViz**, creates a split-screen video:

| **Original (Left)** | **Counterfactual (Right)** |
| :--- | :--- |
| Shows the path the AI actually took. | Shows the "What if?" path. |
| You see the car moving towards the goal. | You might see it hit a wall or take a longer route. |

This visual proof helps us trust (or debug) the AI.
<!-- slide -->
# 4. Scorekeeping (Rewards)
We analyze the "score" for every step:
- **Green Bars**: Good moves (getting closer, dropping off).
- **Red Bars**: Bad moves (hitting walls, wasting time).

If the **Original** path has more Green and less Red than the **Counterfactual** path, the AI made the right choice!
<!-- slide -->
# 5. New Feature: Specific Starting Points
Previously, the simulation started at random spots. 
Now, we can say: 
> *"Start exactly at this complicated intersection (State 123)."*

This lets us test the AI on specific hard problems over and over until we understand its behavior perfectly.
````

## Common Q&A

Here are some questions a non-expert might ask, with simple answers.

### Q1: Why does the taxi sometimes spin in circles?
**A:** The AI is still learning! Just like a student driver, it makes mistakes. COViz helps us see *why*—maybe it thought going straight would lead to a crash (even if it wouldn't), so it turned instead.

### Q2: What do the red and green bars actually measure?
**A:** They measure "Rewards". Think of it like a video game score.
- **+20 points** for a successful drop-off (Green).
- **-1 point** for every second passed (Red) - encouraging speed.
- **-10 points** for hitting a wall (Big Red).

### Q3: Why are there two videos playing side-by-side?
**A:** The left video is **Reality** (what the AI chose). The right video is **Alternative Reality** (what would have happened if it made a different move at the start). It’s like rewinding a movie and choosing a different ending.

### Q4: Can I drive the taxi myself?
**A:** This specific program is for analyzing the *AI's* brain. However, we could technically set it up for a human to control, but the goal here is to audit the computer's logic.

### Q5: What is "State 123"?
**A:** The computer sees the world as a list of numbers. "State 123" is just a shorthand code for "Taxi is at row 2, column 3, passenger is inside, destination is the Hotel." It ensures we are looking at the exact same traffic situation every time.
