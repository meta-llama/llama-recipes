## Quick Web UI for Llama2 Chat
If you prefer to see Llama2 in action in a web UI, instead of the notebooks above, you can try one of the two methods:

### Running [Streamlit](https://streamlit.io/) with Llama2
Open a Terminal, run the following commands:
```
pip install streamlit langchain replicate
git clone https://github.com/facebookresearch/llama-recipes
cd llama-recipes/llama-demo-apps
```

Replace the `<your replicate api token>` in `streamlit_llama2.py` with your API token created [here](https://replicate.com/account/api-tokens) - for more info, see the note [above](#replicate_note).

Then run the command `streamlit run streamlit_llama2.py` and you'll see on your browser the following UI with question and answer - you can enter new text question, click Submit, and see Llama2's answer:

![](../../../docs/images/llama2-streamlit.png)
![](../../../docs/images/llama2-streamlit2.png)

### Running [Gradio](https://www.gradio.app/) with Llama2 (using [Replicate](Llama2_Gradio.ipynb) or [OctoAI](../../llama_api_providers/OctoAI_API_examples/Llama2_Gradio.ipynb))

To see how to query Llama2 and get answers with the Gradio UI both from the notebook and web, just launch the notebook `Llama2_Gradio.ipynb`. For more info, on how to get set up with a token to power these apps, see the note on [Replicate](../../README.md#replicate_note) and [OctoAI](../../README.md##octoai_note).

Then enter your question, click Submit. You'll see in the notebook or a browser with URL http://127.0.0.1:7860 the following UI:

![](../../../docs/images/llama2-gradio.png)
