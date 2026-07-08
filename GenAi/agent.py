class Agent:
    def __init__(
            self,
            name,
            instructions,
            tools=None,
            model="llama-3.3-70b-versatile",
            output_type=None
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = model
        self.output_type = output_type