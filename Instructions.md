See the agenda and based the new method create a Implementation floder where instead of writing code we will implement some ideas from the paper from scracth and make a new Julia library from scracth simmilar to flux but no need for cuda or matrix discrete operations on cpu, we are basically striping out tranistors for a more free and effiencient form of computation, don't copy the exact idea from the paper but instead we take ideas from it and make our own Implementation.

1. The Fundementals

- Implementation/Mathematics/ should contain a few md files explaining the math concept and equations and formulas in the paper

- Implementation/Structure/ covers alterative waves to structure the project like having a math and equations have their own section, or should we build from scarcth or use already decent libaries

- Implementation/Fundementals/ should cover the basic code we should write to get started and the floders get created

2. Wave Interface 

- Here we test 144+ algorithms NONE using cpu or cuda but different waves to interact with hardware waves, audio, emfs, radio, signal processing ect... and save results in log files this is the ONLY time we code

- With our winner wave protocal we make the new markdowns for implementation with no cpu or cuda just pure wave interface

- With The Basics out of the wave and a working signal processing framework we can start to make our basic library fuctions and tools

3. Rise of SOVs Self Oragnizing Vacumm Networks

- We build the basic network library components we need to get started, write the implementation on how to do that

- We build more advanced features and the actual network settings such as optimizers, loss functions, sigmoid, adamw, and the main initial library wave based, add dataset support

- We should be finsihed at this point and have a working julia library replacing MLPs and KANs and saying hello to SOVs (note the paper says c++ and c#, but only use Julia)

4. Backwards Compabilty and model formats

- We don't save models as Bin Pickle ckpt or safetensors or bson even instead we save as an mp4 which the video itself is a sequence of gifs from gif 1 to gif n the first gif representing the firsts few weights of model and the nth representing the final weights of the whole model, keep in mind gifs 1 to n are a whole like a video with n scenes since gifs are short we store the final model in a mp4 we can decode back into a zip floder containing the gifs, each gif 1 to n should model the frequecies and model learning

- Add gguf support to convert the mp4 and gifs model weights to readble python gguf that can be quantized

- Document, Publish, and add future features and updates

This is a no code Explanation and should ONLY code when the step 2 finding optimal wave computing
