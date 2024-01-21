import fire
from lm_eval.base import LM
from lm_eval import tasks, evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HuggingFaceModel(LM):
    def __init__(self, model_name, tokenizer_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def loglikelihood(self, ctx, cont):
        # Encode context and continuation
        input_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
        cont_ids = self.tokenizer.encode(cont, add_special_tokens=False)

        # Concatenate context and continuation
        input_ids += cont_ids

        # Calculate log likelihood
        with torch.no_grad():
            outputs = self.model(input_ids=torch.tensor([input_ids]), labels=torch.tensor([input_ids]))
            log_likelihood = -outputs.loss.item() * len(cont_ids)

        return log_likelihood, len(cont_ids)
    
    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        encoded = encode_tokens(self._tokenizer,
            string, bos=True, eos=False, device=self._device)
        # encoded is a pytorch tensor, but some internal logic in the
        # eval harness expects it to be a list instead
        # TODO: verify this for multi-batch as well
        encoded = encoded.tolist()
        return encoded

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inputss):

        max_new_tokens = 1
       
        logits = model(inputs)
        return logits
    
    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception('unimplemented')
    # Implement other required methods if needed

def evaluate_model(model_name, tokenizer_name, task_list):
    # Instantiate the model
    model = HuggingFaceModel(model_name, tokenizer_name)

    # Convert task_list string to list
    task_list = task_list.split(',')

    # Evaluate
    results = evaluator.evaluate(lm=model, tasks=tasks.get_task_dict(task_list), provide_description=True)
    print(results)

if __name__ == "__main__":
    fire.Fire(evaluate_model)
