# Llama2 Demo Apps 

This folder showcases the Llama2-powered apps.

## HelloLlama - Quickstart in Running Llama2

The first three demo apps show:
* how to run Llama2 locally on a Mac, in the Google Colab notebook, and in the cloud using Replicate;
* how to use [LangChain](https://github.com/langchain-ai/langchain), an open-source framework for building LLM apps, to ask Llama general questions in different ways;
* how to use LangChain to load a recent PDF doc - the Llama2 paper pdf - and ask questions about it. This is the well known RAG (Retrieval Augmented Generation) method to let LLM such as Llama2 be able to answer questions about the data not publicly available when Llama2 was trained, or about your own data. RAG is one way to prevent LLM's hallucination. 
* how to ask follow up questions to Llama by sending previous questions and answers as the context along with the new question, hence performing multi-turn chat or conversation with Llama.

### Running Llama2 locally on Mac
To run Llama2 locally on Mac using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), first open the notebook `HelloLlamaLocal`. Then replace `<path-to-ggml-model-q4_0.gguf>` in the notebook `HelloLlamaLocal` with the path either to your downloaded quantized model file [here](https://drive.google.com/file/d/1afPv3HOy73BE2MoYCgYJvBDeQNa9rZbj/view?usp=sharing), or to the `ggml-model-q4_0.gguf` file built with the following commands:
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python3 -m pip install -r requirements.txt
python convert.py <path_to_your_downloaded_llama-2-13b_model>
./quantize <path_to_your_downloaded_llama-2-13b_model>/ggml-model-f16.gguf <path_to_your_downloaded_llama-2-13b_model>/ggml-model-q4_0.gguf q4_0
```

### Running Llama2 in Google Colab
To run Llama2 in Google Colab using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), click the Colab notebook link [here](https://colab.research.google.com/drive/1-uBXt4L-6HNS2D8Iny2DwUpVS4Ub7jnk?usp=sharing) and download the quantized Llama2-13b-chat model [here](https://drive.google.com/file/d/1afPv3HOy73BE2MoYCgYJvBDeQNa9rZbj/view?usp=sharing) and upload it, as well as the nba.csv file in this repo to your Google drive, so you can access those files in cells #6 and #14. Then run each cell. Notice on the Colab T4 GPU, the inference in cell #18 took more than 20 minnutes to return; running the notebook locally on M1 MBP took about 20 seconds.

### Running Llama2 Hosted in the Cloud
[The Cloud version](HelloLlamaCloud.ipynb) uses LangChain with Llama2 hosted in the cloud on [Replicate](https://replicate.com). The demo shows how to use LangChain to ask Llama2 questions about **unstructured** data stored in a PDF.

[Note on using Replicate](#replicate_note) To run the demo app, you'll need to first sign in with Replicate with your github account, then create a free API token [here](https://replicate.com/account/api-tokens) that you can use for a while. After the free trial ends, you'll need to enter billing info to continue to use Llama2 hosted on Replicate - according to Replicate's [Run time and cost](https://replicate.com/meta/llama-2-13b-chat) for the Llama2-13b-chat model used in our demo apps, the model "costs $0.000725 per second. Predictions typically complete within 10 seconds." This means each call to the Llama2-13b-chat model costs less than $0.01 if the call completes within 10 seconds. If you want absolutely no costs, you can refer to the section "Running Llama2 locally on Mac" above.

## [NBA2023-24](StructuredLlama.ipynb): Ask Llama2 about Structured Data
This demo app shows how to use LangChain and Llama2 to let users ask questions about **structured** data stored in a SQL DB. As the 2023-24 NBA season is around the corner, we use the NBA roster info saved in a SQLite DB to show you how to ask Llama2 questions about your favorite teams or players.

## [VideoSummary](VideoSummary.ipynb): 
This demo app uses Llama2 to return a text summary of a YouTube video.

## [BreakingNews](LiveSearch.ipynb): Ask Llama2 about Live Data
This demo app shows how to perform live data augmented generation tasks with Llama2 and [LlamaIndex](https://github.com/run-llama/llama_index), another leading open-source framework for building LLM apps: it uses the [You.com serarch API](https://documentation.you.com/quickstart) to get breaking news and ask Llama2 about them.

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

![](llama2-streamlit.png)
![](llama2-streamlit2.png)

### Running [Gradio](https://www.gradio.app/) with Llama2

To see how to query Llama2 and get answers with the Gradio UI both from the notebook and web, just launch the notebook `Llama2_Gradio.ipynb`, replace the `<your replicate api token>` with your API token created [here](https://replicate.com/account/api-tokens) - for more info, see the note [above](#replicate_note).

enter your question, click Submit. You'll see in the notebook or a browser with URL http://127.0.0.1:7860 the following UI:

![](llama2-gradio.png)

## LICENSE

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.