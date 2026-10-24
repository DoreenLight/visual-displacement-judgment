# System One models: Jev and its open-source alternatives

> **这是网页快照，不是同行评审论文。**
> 来源：<https://blog.invidelabs.com/system-one-models-jev-laya-open-alternatives/>
> 抓取时间：2026-10-01 20:37
> HTTP 状态：200　｜　原始字节：54,315　｜　SHA256：`da70d8e94782bb0e`
> **用途**：本条在语料四中的证据等级为「社区报告」。引用时必须标注为网页型证据，
> 不得与同行评审论文并列。抓取日期之后页面可能变化，引用前请核对原链接。

**页面描述**：What System One models are, how Jev works, and how Laya, Nimble, Kev and OpenThai compare on limits and test results.


---

System One models: Jev and its open-source alternatives 

 - 

 - 
 - 

 - 

 - 

 - 

- 

 Login 

 Sep 22, 2026

 7 min read

 System One models 

# What are System One models? Jev and its open-source alternatives

 TypeSafe's Jev returns typed answers and probabilities instead of text. Laya, Nimble, Kev and OpenThai now offer open versions of the same idea. This guide covers how they work, their limits and test results, and when a plain classifier is the simpler option.

 Suppose a support system receives a message about a duplicate charge. It needs to choose a queue and determine whether the customer asked for a refund. It also needs to decide whether a person should review the case. A language model can explain those judgments in a paragraph. The support system needs a queue, a refund-request probability, and a review flag. Its code sets that flag using the model's answers and probabilities.

 On September 15, 2026, TypeSafe introduced Jev for that kind of work. TypeSafe calls it the first "System One" model. The name refers to the fast, intuitive "System 1" thinking that Daniel Kahneman described in Thinking, Fast and Slow . TypeSafe uses the term for AI models built to make quick, structured judgments that software can use directly.

 Jev takes supplied information and a set of questions with predefined answer types. It returns typed answers with probabilities instead of writing a reply or deciding the whole workflow. TypeSafe opened Jev in early access at launch.

 Within a week, other teams released open models that take the same kind of request: Laya from ConvAI Innovations, Nimble from Bespoke Labs, Kev from Jared Palmer, and OpenThai-SystemOne from the OpenThai team. These models share a style of input and output, but they do not all use the same architecture or training method, and they break in different places.

## What does a System One model return?

 In TypeSafe's documentation, the state is the material the model should examine. It can be a string or a JSON object containing a ticket, transaction details, and a policy. The application then asks questions and lists the answers each one is allowed to return.

 Jev exposes three question types:

 Question type 
 What the application asks 
 What comes back 

 Choice 
 Which of these queues should receive the ticket? 
 One allowed option and a probability for each option 

 Score 
 How urgent is it on this defined scale? 
 A position on the scale and probabilities across its levels 

 Noul 
 Does the message explicitly request a refund? 
 A probability from 0 to 1 for "yes" 

 Choice and Score also return a confidence value. TypeSafe defines it as a summary of how concentrated the answer's probability distribution is. Noul has no separate confidence field. A confidence of 0.9 does not mean the answer is right 90% of the time.

 The application chooses what information the model sees and which answers it may return. In the duplicate-charge example, the model might judge that the customer requested a refund. Code must still check the transaction records and refund policy before acting. It can send uncertain cases to a person for review. TypeSafe recommends asking several narrow questions and combining their results in code when a judgment has multiple parts.

 The model supplies judgments that software can inspect directly, but it can't generate text. Jev currently accepts text, including JSON-shaped text; it does not take images, audio, or video directly .

## Jev's speed, cost, and accuracy claims

 TypeSafe says Jev uses parallel sampling and a training approach it calls Reinforcement Learning for Calibrated Decisions, or RLCD. Its launch post quotes 70 to 500 milliseconds per request and $0.042 per million input tokens, with output unmetered. TypeSafe says it ran its published evals from laptops on the US West Coast, where the service is based, so latency from elsewhere can be higher.

 The same post reports much larger gains against language models on TypeSafe's own workflow evaluations. The company says its 193.6-times speed and 444.6-times cost results are likely at the high end of real-world gains. It also notes that members of its model team made the evaluation workflows, which could introduce bias. The reference answers are the average of GPT-6 Astra and Fable 5.1, not human labels. The results measure agreement with those two models on TypeSafe's published workflows, not accuracy on your data.

 The claim you can check yourself is the output contract. Jev cannot return a queue name outside the options supplied for a Choice question. Your code never has to handle an invalid Choice label. Jev can still choose the wrong allowed queue, misread a policy, or assign a misleading probability. TypeSafe's launch post charts the schema guarantee as a 0% hallucination rate. The post says that figure isn't measured; it follows from schema matching.

 An LLM with structured output also returns something your code can parse, so the output format alone doesn't separate the two. Compare them on whether the answer is right, whether the probability tracks how often it's right, and what each request costs in time and money.

## Laya and the open responses

 Laya is an open-weight family released in the days after Jev. Its model card lists Apache-2.0 weights and several checkpoints. The English model uses ModernBERT-large with a decision head; a multilingual checkpoint uses mmBERT-base. The maker also publishes a specialized typed-decisions checkpoint. Unlike Jev, Laya runs on your own hardware. The English checkpoint is about an 808 MB download.

 Laya's published implementation uses bidirectional encoders and scores candidate answers at option markers. Its maker also calls its training approach RLCD, the name TypeSafe uses for Jev's training. TypeSafe hasn't published its method, so there's no way to tell how close the two are. Laya's maker reports 32.8 milliseconds for one question on the multilingual checkpoint and 39.5 milliseconds on the English checkpoint, both on a Tesla T4.

 Laya's maker also publishes its limits. Choice accuracy falls when the list grows beyond about 20 options in one of its stress tests. The offered options share a fixed token budget, so long lists leave little room to describe each one. The maker recommends fewer choices or a two-stage hierarchy. Its strong result on a typed-decisions benchmark comes from a checkpoint fine-tuned on that benchmark's training split; the maker says an out-of-the-box base model scores much lower. On Khmer, the English checkpoint got every answer wrong while reporting 0.952 confidence. That's why Laya ships a router that picks a checkpoint by script before the model runs.

 An outside evaluation ran laya 0.3.4 and the multilingual checkpoint on 100 hand-written English scenarios. The author deliberately made them adversarial, with negation, retractions, buried evidence, and other traps. The run was zero-shot, without fine-tuning or temperature fitting. The author cautions that its 40.3% accuracy is not the figure to expect on ordinary inbound mail. On this set, a constant answer per question that read none of the input reached 53.1%. The evaluator also found that 54% of answers with confidence at least 0.90 were wrong, and shuffling the order of Choice options changed about one answer in five. It tested one checkpoint against one author's labels. Its value is the failure list: negation, urgency judged by tone rather than facts, and answers that change with option order.

 Other releases take different routes:

 Model 
 Published approach 
 What to keep in mind 

 Bespoke Nimble 
 An open Qwen3.5-9B adaptation with data, training recipe, and typed-decision interface 
 In Bespoke's 324-example holdout, Nimble matched 292 synthetic reference labels and Jev 1.13.0 matched 302. The examples cover six source families, and no person reviewed the labels. 

 Kev 
 A family of open Qwen-based models using LoRA adapters and a small answer-scoring head, served through a Jev-shaped local API 
 The current repository lists 0.8B, 4B, and 9B Qwen3.5 models, along with earlier checkpoints. Checkpoint and hardware matter in its own latency comparisons. 

 OpenThai-SystemOne 
 Apache-2.0 weights and code for a Thai-and-English decision model, announced September 20 
 The team's v0.3 uses a Qwen3.5-0.8B base. Its quality and latency figures remain maker results. 

 Nimble, Kev, and OpenThai build on Qwen decoder models; Laya uses encoders. Their training data, context limits, and hardware needs differ too.

 They also cap answer choices differently. TypeSafe says Jev supports up to 255 options for a Choice question. Kev documents 1 to 255, while Nimble allows at most 26 string choices for an enum field. Laya's published stress test shows a quality drop above about 20 choices with its default option budget. A model that accepts 255 options can still choose badly from a long list, so test with the number of options you'll actually use.

## Does this replace an LLM or an ordinary classifier?

 These models suit apps that make many small decisions about incoming text and know the possible answers in advance. Ticket routing, moderation triage, document tagging, relevance scoring, and checking whether a message contains a specific request are examples. A generative model remains useful when an application needs to write a reply, explain a decision in natural language, or plan multi-step work.

 Compare against a plain classifier too. If a team has a stable set of labels and enough representative examples, a conventional classifier or a few rules can be simpler and cheaper. A question in the Jev launch discussion raised this directly for a fraud dataset. A general decision model earns its place when the questions or choices change per request and training a classifier for each isn't practical. If your labels are stable, run your existing classifier against it before switching.

 The "System One" name should not be taken to mean that models making non-generative decisions began in September 2026. Laya's maker, for example, points to a March 2025 paper about predicting sales-conversation outcomes with reinforcement learning. That paper predicts one thing, whether a sales conversation converts, rather than answering arbitrary questions. Until TypeSafe publishes its method, nobody can say how close Jev is to it.

## How to evaluate one for your application

 Start with the decision your code actually needs. Define the allowed answers and gather examples from the environment where the model will run. Include ambiguous tickets, missing context, long inputs, unfamiliar language, and cases where none of the offered labels fits. TypeSafe's question guidance recommends adding an other or none of the above choice when the options may be incomplete.

 Then compare Jev, an open model, your present LLM path, and any existing rules or classifier on the same cases. Measure the selected answer's accuracy and the quality of the probabilities separately. Check how often a threshold sends cases to review and how often it lets a wrong decision through. Keep the checkpoint and question wording fixed while comparing, and record the hardware and full request latency. For self-hosted models, count GPU time, memory, and cold starts. There's no per-token bill, but inference still costs money.

 For a first deployment, let the model suggest a route and send uncertain cases to a person. On Bespoke's narrow holdout, Nimble came close to Jev; the adversarial Laya test shows why a multilingual checkpoint needs a separate zero-shot evaluation before it handles decisions automatically. Expand the model's role only after it clears the error rate your own workflow can tolerate.

### Member discussion