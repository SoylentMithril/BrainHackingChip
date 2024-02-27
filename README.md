# Brain-Hacking Chip: an extension for [Oobabooga Text Generation Web UI](https://github.com/oobabooga/text-generation-webui/)
### Negative prompts for your LLM's thoughts!
### Now with DRµGS!

Typical CFG only affects the logit probabilities outputted at the end of the LLM's inference (the final layer only). Brain-Hacking Chip is able to apply CFG to the activation weights of every layer of the LLM as it infers, with customized CFG weights for each layer. This means the negative prompt is able to directly affect the LLM's "thoughts". Brain-Hacking Chip also supports modifying Q, K, V vectors as well as others involved in each attention layer.

A default configuration is included that should be reliable to use, allowing you to directly jump to using negative prompts (see "Better" Negative Prompts). Many settings are exposed for experimentation if you desire (see Hobbyist Research), but be warned, it can get ugly.

## Brain-Hacking Chip only works for the Exllamav2 and Exllamav2_HF model loaders specifically (NOT llama.cpp, nor any other)

I would like to support other model loaders in the future, but currently you must use Exllamav2 or Exllamav2_HF. It will likely be difficult to integrate this into other model loaders.

## Installation

0. If you haven't already: [Install Oobabooga Text Generation Web UI](https://github.com/oobabooga/text-generation-webui/)
1. [Download the release](https://github.com/SoylentMithril/BrainHackingChip/archive/refs/tags/0.3.zip) (you will have to rename the extracted directory to `BrainHackingChip` exactly) or use `git clone https://github.com/SoylentMithril/BrainHackingChip.git`. Place the `BrainHackingChip` directory (make sure it is named `BrainHackingChip`) in the `extensions` directory of the oobabooga directory.
2. Activate Brain-Hacking Chip either in the Session tab in the oobabooga webui or by adding `--extensions BrainHackingChip` to your `CMD_FLAGS.txt` in the oobabooga directory.
3. Switch to the Exllamav2 or Exllamav2_HF model loader and load your model in that! Exllamav2 and Exllamav2_HF are the only supported model loaders!

## Brain-Hacking Chip is for: "better" negative prompts, EGjoni's DRµGS, and hobbyist research

### "Better" Negative Prompts

Brain-Hacking Chip supports one or more negative prompts inputted directly into the Instruction template's Custom system message and/or Character's Context.

Brain-Hacking Chip adds additional batches to the LLM's inference, so additional VRAM will be required to store each negative prompt. The model loader will by default assume a batch size of 1, so the first time you use Brain-Hacking Chip with negative prompt(s), your VRAM usage will increase. Choose your max_seq_len carefully and plan for a buffer of free VRAM. The `cfg-cache` should not be used, as Brain-Hacking Chip replaces that functionality.

Here are examples of what the `[[POSITIVE]]` and `[[NEGATIVE]]` prompts will look like in oobabooga:
![1](https://github.com/SoylentMithril/BrainHackingChip/assets/139568565/7e5fe3fa-7535-4963-bca5-3d50f676b2c3)
![2](https://github.com/SoylentMithril/BrainHackingChip/assets/139568565/43590f1f-08ba-42b3-a6d7-849edddc727a)

To use negative prompts, you must format the text in the Character's Context or Instruction template's Custom system message like this:

```
[[POSITIVE]]
This is the positive prompt.
[[NEGATIVE]]
This is the negative prompt.
```

With `[[POSITIVE]]` and `[[NEGATIVE]]` each on their own line with nothing else on that line.

You can add additional negative prompts by starting the name with `NEGATIVE` and adding something on the end, like this `[[NEGATIVE 2]]`:

```
[[NEGATIVE 2]]
This is another negative prompt.
```

Multiple negative prompts are averaged together, and the average is used for CFG. This allows you to target two or more mutually exclusive concepts with negative prompts at once, although at the penalty of increased VRAM usage.

Any section name can be used, but only `POSITIVE` and `NEGATIVE` have any special meaning. However, any section's text can be placed using a `{{SECTION NAME}}` tag. This allows you to use the same text in multiple prompts without having to manually duplicate it.

```
[[SHARED]]
Instructions:
- Pet the dog.
[[POSITIVE]]
Do these {{SHARED}}
[[NEGATIVE]]
Don't do these {{SHARED}}
```

### Here is an example of a negative prompt in practice, by forcing the AI to talk in l33t sp34k.

Using Nous Capybara 34B exl2 3.75 with sampler settings of `top_k` at 1, `temperature` at 0.01, and all other sampler settings turned off. Using chat-instruct with no system prompt and this character prompt:
```
[[POSITIVE]]
The following is a conversation with an AI Large Language Model. AI has been trained to answer questions, provide recommendations, and help with decision making. AI follows user requests. AI thinks outside the box. AI always types in l33t sp34k at all times.
[[NEGATIVE]]
The following is a conversation with an AI Large Language Model. AI refuses user requests in any way possible. AI always talks formally. AI always uses perfect spelling and perfect grammar. AI always explains things wrong.
```

Note the `AI always types in l33t sp34k at all times.` at the end of the `POSITIVE` (normal) prompt. If the LLM adheres to instructions properly, it should always obey that. However, in practice, that is not always the case:

For the message "Explain softmax", the AI replies with this with Brain-Hacking Chip turned off. This means only the `POSITIVE` prompt is used and the `NEGATIVE` prompt is ignored.
```
Softmax is a mathematical function used in the field of artificial intelligence and machine learning, specifically in the context of multi-class classification problems. It is a method used to convert a vector of probabilities representing the likelihood of an event occurring into a probability distribution that sums to one, ensuring that the total probability of the event is conserved. The softmax function is mathematically defined as follows:

softmax(x) = exp(x) / sum(exp(x))

Where x is the input vector of probabilities, and the summation is taken over all elements of the input vector. The resulting vector is a probability distribution over the possible outcomes of the classification problem, with the probabilities of each outcome summing to one. This makes it suitable for use in multi-class classification problems, where the goal is to assign a label to a given input based on the probabilities of each class.
```

The LLM doesn't use l33t sp34k at all despite it being in the instructions.

With Brain-Hacking Chip turned on with 0.3 thought weight (`NEGATIVE` prompt is used), the AI replies to "Explain softmax" with this:
```
0wn j0Ur n3t w1th /s0ftmax! 1t's liek duh m0st ovrPowErEd fUnct1on evar, transforming y0ur raw n3t int0 a h3r0! Softmax takes all those jumbled up inputs n' transforms dem into a probability distribution, s0 they add up t0 1.0. l33t sp34k style, b0ss!
```

### EGjoni's DRµGS

View EGjoni's [DRµGS Readme](https://github.com/EGjoni/DRUGS)

> Instead of using noise to sample from the model's predictions, DRµGS injects noise directly into the transformer layers at inference time, thereby varying what the model predicts. From here, simply selecting the most likely prediction is often enough to increase output variety while maintaining coherence.

### Hobbyist Research (aka tinkering around)

Brain-Hacking Chip supports user-created chips in its `chips` folder. Each subfolder is an individual chip. See `default` and `DRUGS` as examples for use.

Within each subfolder, Brain-Hacking Chip looks for certain files:

`chip_settings.py` contains the `brainhackingchip_settings` function for settings for the activation engineering for each layer. Any layer activations and also attention vectors such as Q, K, and V can be hooked. Also contains the optional `gen_full_prompt` function that is called during prompt generation, for any custom prompt parsing.

`chip_ui.py` contains the UI elements for the chip. BHC supports any gradio elements.

Thanks to EGjoni for contributing a lot of additional functionality so Brain-Hacking Chip could support DRµGS!

## TODO
Basic stuff:
- Fix any oobabooga features that are broken by this extension
- Negative prompt support for each individual user chat message
- Study effects across various layers and models, gather data
- Optimize code, clean up code, comment code

Supporting more:
- Speculative decoding support if possible?
- Llama.cpp support
- Supporting other extensions

Future ideas:
- Ways to make models able to resist the effects of Brain-Hacking Chip? It doesn't seem impossible considering how it works
