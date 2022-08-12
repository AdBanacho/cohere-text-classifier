import cohere
import prompt
from loadExamples import examples
from cohere.classify import Example

class CoHere:
    def __init__(self, api_key):
        self.co = cohere.Client(f'{api_key}', '2021-11-08')
        self.examples = []

    def list_of_examples(self):
        for e in examples():
            self.examples.append(Example(text=e[0], label=e[1]))

    def cohere(self, question):
        return self.co.generate(
              model='medium',
              prompt=prompt.prompt_Steven_qa(question),
              max_tokens=50,
              temperature=1,
              k=0,
              p=0.75).generations[0].text

    def classify(self, inputs):
        return self.co.classify(
            model='medium',
            taskDescription='',
            outputIndicator='',
            inputs=inputs,
            examples=self.examples
        ).classifications

