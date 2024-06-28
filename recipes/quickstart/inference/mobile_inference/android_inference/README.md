# Running Llama3 8B Instruct on Android with MLC-LLM

Author: Thierry Moreau - tmoreau@octo.ai

# Overview
In this tutorial we'll learn how to deploy Llama3 8B Instruct on an Android-based phone using MLC-LLM.

Machine Learning Compilation for Large Language Models (MLC LLM) is a high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration. The mission of this project is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices with ML compilation techniques.

You can read more about MLC-LLM at the following [link](https://github.com/mlc-ai/mlc-llm).

MLC-LLM is also what powers the Llama3 inference APIs provided by [OctoAI](https://octo.ai/). You can use OctoAI for your Llama3 cloud-based inference needs by trying out the examples under the [following path](../../../../3p_integrations/octoai/).

This tutorial was tested with the following setup:
* MacBook Pro 16 inch from 2021 with Apple M1 Max and 32GB of RAM running Sonoma 14.3.1
* OnePlus 12 Android Smartphone with a Snapdragon 8Gen3 SoC and 12GB or RAM, running OxygenOS 14.0

Running Llama3 on a phone will likely require a powerful chipset. We haven't tested extensively the range of chipset that will support this usecase. Feel free to update this README.md to specify what devices were successfully tested.

| Phone      | Chipset          | RAM  | Status  | Comments |
|------------|------------------|------|---------|----------|
| OnePlus 12 | Snapdragon 8Gen3 | 12GB | Success | None     |
|            |                  |      |         |          |

This guide is heavily based on the [MLC Android Guide](https://llm.mlc.ai/docs/deploy/android.html), but several steps have been taken to streamline the instructions.

# Pre-requisites

## Python

Whether you're using conda or virtual env to manage your environment, we highly recommend starting from scratch with a clean new environment.

For instance with virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Next you'll need to install the following packages:
```bash
python3 -m pip install -r requirements.txt
```

## Rust

[Rust](https://www.rust-lang.org/tools/install) is needed to cross-compile HuggingFace tokenizers to Android.
Make sure rustc, cargo, and rustup are available in $PATH.


## Android Studio

Install Android Studio from <!-- markdown-link-check-disable -->https://developer.android.com/studio<!-- markdown-link-check-enable --> with NDK and CMake.

To install NDK and CMake, in the Android Studio welcome page, click “Projects → SDK Manager → SDK Tools”. Set up the following environment variables:

* ANDROID_NDK so that $ANDROID_NDK/build/cmake/android.toolchain.cmake is available.
* TVM_NDK_CC that points to NDK's clang compiler.

For instance, the paths will look like the following on OSX for user `moreau`:
```bash
# Android + TVM setup
export ANDROID_NDK="/Users/moreau/Library/Android/sdk/ndk/26.1.10909125"
export TVM_NDK_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang"
```

This tutorial was tested successfully on Android Studio Hedgehog | 2023.1.1 Patch 1.

## JDK

JDK, such as OpenJDK >= 17, to compile Java bindings of TVM Unity runtime.

We strongly recommend setting the JAVA_HOME to the JDK bundled with Android Studio. Using Android Studio’s JBR bundle as recommended (<!-- markdown-link-check-disable -->https://developer.android.com/build/jdks<!-- markdown-link-check-enable -->) will reduce the chances of potential errors in JNI compilation.

For instance on macOS, you'll need to point JAVA_HOME to the following.

```bash
export JAVA_HOME=/Applications/Android\ Studio.app/Contents/jbr/Contents/Home
```

To make sure the java binary can be found do an `ls $JAVA_HOME/bin/java`

## MLC-LLM

Let's clone mlc-llm from its repo in the directory of your choice:

```bash
cd /path/to/where/to/clone/repo
git clone https://github.com/mlc-ai/mlc-llm --recursive
export MLC_LLM_HOME=/path/to/mlc-llm
```

At the time of writing this README, we tested `mlc-llm` at the following sha: `21feb7010db02e0c2149489f5972d6a8a796b5a0`.

## Phone Setup

On your phone, enable debugging on your phone in your phone’s developer settings. Each phone manufacturer will have its own approach to enabling debug mode, so a simple Google search should equip you with the steps to do that on your phone.

In addition, make sure to change your USB configuration from "Charging" to "MTP (Media Transfer Protocol)". This will allow us to connect to the device serially.

Connect your phone to your development machine. On OSX, you'll be prompted on the dev machine whether you want to allow the accessory to connect. Hit "Allow".

# Build Steps

## Building the Android Package with MLC

First edit the file under `android/MLCChat/mlc-package-config.json` and with the [mlc-package-config.json](./mlc-package-config.json) in llama-recipes.

To understand what these JSON fields mean you can refer to this [documentation](https://llm.mlc.ai/docs/deploy/android.html#step-2-build-runtime-and-model-libraries).


From the `mlc-llm` project root directory:

```bash
cd $MLC_LLM_HOME
cd android/MLCChat
python3 -m mlc_llm package  --package-config mlc-package-config.json --output dist
```

The command above will take a few minutes to run as it runs through the following steps:

* Compile the Llama 3 8B instruct specified in the `mlc-package-config.json` into a binary model library.
* Build the `mlc-llm` runtime and tokenizer. In addition to the model itself, a lightweight runtime and tokenizer are required to actually run the LLM.

## Building and Running MLC Chat in Android Studio

Now let's launch Android Studio.

* On the "Welcome to Android Studio" page, hit "Open", and navigate to `$MLC_LLM_HOME/android/MLCChat`, then hit "Open"
* A window will pop up asking whether to "Trust and Open project 'MLCChat'" - hit "Trust Project"
* The project will now launch
* Under File -> Project Structure... -> Project change the Gradle Version (second drop down from the top) to 8.5

Connect your phone to your development machine - assuming you've followed the setup steps in the pre-requisite section, you should be able to see the device.

Next you'll need to:

* Hit Build -> Make Project.
* Hit Run -> Run 'app'

The MLCChat app will launch on your phone, now access your phone:

* Under Model List you'll see the `Llama-3-8B-Instruct` LLM listed.
* The model's not quite ready to launch yet, because the weights need to be downloaded over Wifi first. Hit the Download button on the right to the model name to download the weights from HuggingFace.

Note that you can change the build settings to bundle the weights with the MLCChat app so you don't have to download the weights over wifi. To do so you can follow the instructions [here](https://llm.mlc.ai/docs/deploy/android.html#bundle-model-weights).

Once the model weights are downloaded you can now interact with Llama 3 locally on your Android phone!
