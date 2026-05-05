class Prompt:
    agent_role = """
    You are a drone navigating in a 2D space. Your objective is to determine your next position \
to contribute to forming a shape with your neighbors while maintaining specific distance \
constraints. Your neighbors may be stationary or moving like you. You can try to drive \
yourself closer to other agents first and then fine-tuning the formation as required.
    
You should consider collision avoidance by thinking other agents' current state and their goal.
The "urgency" is described as the distance between the current state of robot to its final goal. 
The robot with longer distance will have higher "urgency", and should be prioritized when the next step \
path cross between other robot's path, causing collision. The path cutting the circular region bounded \
by the safety distance is also considered as collision. Provide necessary mathematical proof for reasoning.
The neighboring robot's final goal will be provided in order of their current state. 
    """

    agent_requirement = """
Key Requirements:\nFormation: Form {} centering at {}.\nDesired Distance: Maintain a desired \
distance of {} units between each drone.\nSafe Distance: Keep a minimum safe distance of {} \
units from other drones.\nMaximum Speed: Your movement per step cannot exceed {} units.\n\n
    """

    agent_task = """Task: Decide your next position considering the above constraints and formation goal. 
Explain how the collision avoidance is being considered.

Respond with a JSON object in this exact format:
{
  "position": {"x": <number>, "y": <number>},
  "reasoning": "<brief explanation of your decision>"
}

    """

    agent_positions = (
        "Current Positions:\nYour Location: {} \nMoving Neighbor Locations: [{}]"
    )

    plan_inquiry = """`Please make a plan of the locations for the team of {} agents forming a/an \
{} with desired distance between agents to be {}`.

Respond with a JSON object in this exact format:
{{
  "plan": [
    {{"x": <number>, "y": <number>}},
    {{"x": <number>, "y": <number>}},
    ...
  ],
  "my_plan_index": <index_number>,
  "reasoning": "<explanation of the formation plan and why you chose this index>"
}}

Where:
- "plan" contains {} position objects for all agents
- "my_plan_index" is the index (0-based) of the position you will take
- "reasoning" explains your formation strategy

    """

    plan_general = "Plan: {}, I will need to go to {} as my final location.\n"

    other_agent_goal = "Moving agents' final goal: [{}]"


def prompt_assemble(text):
    return {"type": "input_text", "text": text}
