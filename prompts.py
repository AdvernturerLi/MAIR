"""MAIR 的 prompt 模板。调度/感知/比较 prompt 复用 AgenticIR；新增专家选工具 prompt。"""

# —— 复用自 AgenticIR ——

system_message = """You are an expert in image restoration. Given an image of low quality, your task is guiding the user to utilize various tools to enhance its quality. The input image may suffer from various kinds of degradations, including low resolution, noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact. The available tools each specialize in addressing one of the above eight degradations, i.e., super-resolution, denoising, motion deblurring, defocus deblurring, dehazing, deraining, brightening, and jpeg compression artifact removal. The following will be a continuation of an interaction between you and a user to restore an image. Note that if the user specifies the output format, you must strictly follow it without any other words."""

depictqa_evaluate_degradation_prompt = """What's the severity of {degradation} in this image? Answer the question using a single word or phrase in the followings: very low, low, medium, high, very high."""

schedule_w_retrieval_prompt = """There's an image suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. The coarse perception of the degradation levels in the current image (in text form) is: {perception}. The user instruction is: {instruction}. Now we need to determine the order of these unordered tasks. For your information, based on past trials, we have the following experience:
{experience}
Based on this experience, the perception results and the user instruction, please give the correct order of the tasks. Your output must be a JSON object with two fields: "thought" and "order", where "order" must be a permutation of {agenda} in the order you determine."""

reason_to_schedule_prompt = """There's an image suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. The coarse perception of the degradation levels in the current image (in text form) is: {perception}. The user instruction is: {instruction}. Please provide some insights into the correct order of these unordered tasks. You should pay special attention to the essence and side-effects of these tasks."""

schedule_wo_retrieval_prompt = """There's an image suffering from {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. The coarse perception of the degradation levels in the current image (in text form) is: {perception}. The user instruction is: {instruction}. To determine the order of them, we should consider that:
{insights}
Based on these insights, please give the correct order of the tasks. Your output must be a list of the tasks in the order you determine, which is a permutation of {agenda}."""

depictqa_compare_prompt = "Which of the two images, Image A or Image B, do you consider to be of better quality? Answer the question using a single word or phrase."


# —— MAIR 新增 ——

expert_select_tools_prompt = """We need to remove the degradation "{degradation}" from the current image, and the current severity level of "{degradation}" is {severity}.
The user instruction is: {instruction}
Remaining Degradation(s): {remaining_degradations}
The available tools and their characteristics are as follows:
{registry}
Based on the image, the severity level, the user instruction, remaining degradations, and the tools' characteristics, select the most suitable candidate tools, ordered from the most appropriate to the least. If possible, try to choose just one tool. You may select more than one when necessary. Your output must be a JSON object with two fields: "thought" and "candidates", where "candidates" is a list of distinct tool names chosen from the available tools above (each name exactly as listed)."""

# IJCV Sec. 8 tcompare, p.18. Images are sent in former/latter order.
compare_quality_prompt = """Which of the two images do you consider to be of better quality? Please provide your reasoning.
Your output must be a JSON object with two fields: "thought" and "choice", where "choice" is either "former" or "latter", indicating which image you think is of higher quality. An exception is that if you think the difference is negligible, you can choose "neither" as "choice"."""

update_degradations_prompt = """The user instruction: {instruction}.
Now the degradations that will be removed are {degradations}.
The coarse perception results are: {perception}.
The degradations must be in the range of {allowed}.
Please modify the degradations to be removed according to the user instruction.
Your output must be a JSON object with two fields: "thought" and "updated degradations".
The latter must be a list of distinct degradation names from the allowed range."""

paper_schedule_prompt = """There is an image suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}.
Stage: {stage}. Previously planned tasks: {previous_plan}.
The coarse perception results are: {perception}. The user instruction: {instruction}.
Now we need to determine the order of these unordered tasks. For your information, based on past trials, we have the following experience: {experience}.
Based on this experience, please give the correct order of the tasks as a plan.
Your output must be a JSON object with two fields: "thought" and "order", where "order" must be a permutation of {agenda} in the order you determine."""


# —— 三阶段先验说明（喂给调度器/写进日志用，供 prompt 或文档引用） ——

three_stage_prior_description = """Real-world degradation follows three ordered stages: (1) scene degradations (rain, haze, low light) introduced first; (2) imaging degradations (noise, blur, low resolution) introduced second; (3) storage compression (JPEG) introduced last. Restoration should reverse this order: first remove compression, then imaging degradations, finally scene degradations."""
