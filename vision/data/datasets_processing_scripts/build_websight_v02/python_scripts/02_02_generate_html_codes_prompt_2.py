# [MODULE DOC] Script (WebSight v0.2 step 2, prompt 2): variant of 02_01 that asks
# deepseek-coder-33b-instruct for a
# "beautiful table with many rows" (no images) about each business, written inside a <body> tag.
# Ideas are first reduced to their business part (text before ":"). Argument: sys.argv[1] = NAME_FILE;
# output goes to .../websight_v02_generated_html_codes/prompt_2/<NAME_FILE>.json.
# When run as main: builds GenerationHTMLCodes from the module constants and calls it.
import json
import random
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


NAME_FILE = sys.argv[1]

PATH_IDEAS_WEBSITES = f"/fsx/hugo/ideas_mistral_websight_v02/ideas_1500_per_file/{NAME_FILE}.txt"

MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns ="
    " namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n       "
    " {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are"
    " a powerful AI programming assistant, able to generate HTML and Tailwind CSS codes to create"
    " beautiful websites.\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system'"
    " %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'###"
    " Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content']"
    " + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt"
    " %}\n{{'### Response:'}}\n{% endif %}"
)

BATCH_SIZE = 10
PROMPT = """Code a beautiful table with many rows in HTML and Tailwind CSS for a website about a {idea}. Write the code inside a tag <body>. Don't include any images."""
MAX_NEW_TOKENS = 2048
PAD_TOKEN_ID = 32014
EOS_TOKEN_ID = 32021

TEMPLATE_HTML_CODE = """<html>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
{body_html_code}
</html>"""

OVERUSED_COLOR = "blue"
LIST_COLORS = ["gray", "red", "yellow", "green", "blue", "indigo", "purple", "pink"]

PATH_SAVE_IDEAS_AND_HTML_CODES = f"/fsx/hugo/websight_v02_generated_html_codes/prompt_2/{NAME_FILE}.json"


class GenerationHTMLCodes:
    # [CLASS DOC] Same generator as in 02_01 (load ideas, GPU model, batched greedy generation, HTML
    # extraction,
    # color diversification, JSON save) with an extra idea-preprocessing step.
    def __init__(
        self,
        path_ideas_websites,
        model_name,
        chat_template,
        batch_size,
        prompt,
        max_new_tokens,
        pad_token_id,
        eos_token_id,
        path_save_ideas_and_html_codes,
    ):
        # [FUNCTION DOC] Stores the configuration (ideas path, model name, chat template, batch size, prompt
        # format string with
        # {idea}, max_new_tokens, pad/eos token ids, output JSON path). No I/O.
        self.path_ideas_websites = path_ideas_websites
        self.model_name = model_name
        self.chat_template = chat_template
        self.batch_size = batch_size
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.path_save_ideas_and_html_codes = path_save_ideas_and_html_codes

    def __call__(self):
        # [FUNCTION DOC] Runs load_ideas_websites, process_ideas_websites, load_model_and_tokenizer,
        # all_model_generations and a
        # final save_ideas_and_html_codes.
        self.load_ideas_websites()
        self.process_ideas_websites()
        self.load_model_and_tokenizer()
        self.all_model_generations()
        self.save_ideas_and_html_codes()

    def load_ideas_websites(self):
        # [FUNCTION DOC] Reads the ideas file into self.ideas_websites (list of lines).
        with open(self.path_ideas_websites, "r") as f:
            self.ideas_websites = f.read().split("\n")

    def process_ideas_websites(self):
        # [FUNCTION DOC] Keeps only ideas containing exactly one ":" and replaces each with its lowercased
        # text before the ":"
        # (the business description, dropping the layout details). Mutates self.ideas_websites.
        """Keep only the main business idea, remove the details about the design of the website"""
        self.ideas_websites = [idea.split(":")[0].lower() for idea in self.ideas_websites if len(idea.split(":")) == 2]

    def load_model_and_tokenizer(self):
        # [FUNCTION DOC] Loads the model (flash-attention 2, trust_remote_code, bfloat16) on CUDA and the
        # tokenizer with the custom
        # chat template; sets self.model and self.tokenizer.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_flash_attention_2=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.chat_template = self.chat_template

    def batch_model_generations(self, batch_ideas):
        # [FUNCTION DOC] Same as in 02_01: chat-template each idea, left-pad with PAD_TOKEN_ID, greedy
        # generate, strip prompt
        # tokens, decode without special tokens and eos string. Returns list of str aligned with batch_ideas.
        # KNOWN ISSUE: uses module globals PROMPT / PAD_TOKEN_ID instead of self.prompt / self.pad_token_id,
        # and no
        # attention_mask is passed for the left-padded batch.
        all_messages = [
            [
                {"role": "user", "content": PROMPT.format(idea=idea)},
            ]
            for idea in batch_ideas
        ]
        all_encodeds = [
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            for messages in all_messages
        ]
        len_prompts = [encodeds.shape[1] for encodeds in all_encodeds]
        max_len_prompt = max(len_prompts)
        padded_all_encodeds = torch.full((len(batch_ideas), max_len_prompt), PAD_TOKEN_ID)
        for idx_enc, encodeds in enumerate(all_encodeds):
            padded_all_encodeds[idx_enc, max_len_prompt - len_prompts[idx_enc] :] = encodeds[0]
        model_inputs = padded_all_encodeds.to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generated_ids = generated_ids[:, max_len_prompt:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [el.replace(self.tokenizer.eos_token, "") for el in decoded]
        return decoded

    def extract_html_code_from_generation(self, generation):
        # [FUNCTION DOC] Returns None unless the generation has exactly one "<body " and one "</body>";
        # otherwise wraps the
        # "<body ...</body>" substring in TEMPLATE_HTML_CODE (Tailwind CDN link) and returns it.
        if (generation.count("<body ") != 1) or (generation.count("</body>") != 1):
            return None
        else:
            pos_start_html_code = generation.find("<body ")
            pos_end_html_code = pos_start_html_code + generation[pos_start_html_code:].find("</body>") + len("</body>")
            body_html_code = generation[pos_start_html_code:pos_end_html_code]
            html_code = TEMPLATE_HTML_CODE.format(body_html_code=body_html_code)
            return html_code

    def replace_overused_color_from_html_code(self, html_code):
        # [FUNCTION DOC] Replaces every "blue" substring in html_code by a random color from LIST_COLORS;
        # returns the new string.
        random_color = random.choice(LIST_COLORS)
        html_code = html_code.replace(OVERUSED_COLOR, random_color)
        return html_code

    def save_ideas_and_html_codes(self):
        # [FUNCTION DOC] Dumps self.ideas_and_html_codes ([idea, html] pairs) to the output JSON (overwrite)
        # and prints the count.
        with open(self.path_save_ideas_and_html_codes, "w") as f:
            json.dump(self.ideas_and_html_codes, f)
        print(f"Number of HTML codes: {len(self.ideas_and_html_codes)}")

    def all_model_generations(self):
        # [FUNCTION DOC] Batched loop over ideas (tqdm): generate, extract, recolor; collects successful
        # [idea, html] pairs into
        # self.ideas_and_html_codes and checkpoints to JSON every 10 batches.
        self.ideas_and_html_codes = []
        for idx in tqdm(range(0, len(self.ideas_websites), self.batch_size)):
            batch_ideas = self.ideas_websites[idx : idx + self.batch_size]
            batch_generated_html_codes = self.batch_model_generations(batch_ideas=batch_ideas)
            for idea, generation in zip(batch_ideas, batch_generated_html_codes):
                html_code = self.extract_html_code_from_generation(generation=generation)
                if html_code is not None:
                    html_code = self.replace_overused_color_from_html_code(html_code=html_code)
                    self.ideas_and_html_codes.append([idea, html_code])
            if idx % (10 * self.batch_size) == 0:
                print(f"Saving for idx {idx}")
                self.save_ideas_and_html_codes()


if __name__ == "__main__":
    generation_html_codes = GenerationHTMLCodes(
        path_ideas_websites=PATH_IDEAS_WEBSITES,
        model_name=MODEL_NAME,
        chat_template=CHAT_TEMPLATE,
        batch_size=BATCH_SIZE,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=PAD_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        path_save_ideas_and_html_codes=PATH_SAVE_IDEAS_AND_HTML_CODES,
    )
    generation_html_codes()
