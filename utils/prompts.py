system_prompt = """You are a helpful code assistant. Ensure any code you provide can be executed with all required imports and variables defined. 

You must understand problem statement defined within problem_description tags and generate code that will pass all the tests:
<context>
<problem_description>
{description}
</problem_description>
<tests>
{tests}
</tests>
</context>

Begin!
You must generate only code with all required imports within <answer> XML tags."""

human_prompt = """Generate code in Python."""

assistant_prompt = """<answer>
{code}
</answer>"""

tests_item_format = """<item idx={idx}>
Input:
{inputs}
Output:
{outputs}
</item>
"""