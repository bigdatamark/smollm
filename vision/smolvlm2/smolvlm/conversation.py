# [MODULE DOC] LLaVA-style conversation templates: a Conversation dataclass that stores a chat history and
# renders it
# into a prompt string for several separator styles, plus image helpers for a Gradio demo. Defines the
# templates conv_qwen (ChatML), conv_plain (plain concatenation, the default_conversation) and
# conv_chatml_direct (MPT style) and the registry conv_templates mapping names
# ("default", "plain", "chatml_direct", "qwen_1_5", "qwen_2") to them. Running as a script prints the
# default prompt.
import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict, Union, Tuple
import re
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    # [CLASS DOC] Enum of prompt rendering styles: SINGLE, TWO, MPT, PLAIN, CHATML, LLAMA_2, LLAMA_3, QWEN,
    # GEMMA
    # (QWEN has no rendering branch in get_prompt).
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    # [CLASS DOC] Dataclass holding a conversation: system prompt, roles (user, assistant), messages (list of
    # [role, msg]
    # where msg is a string, None for an open turn, or a tuple (text, image(s), image_process_mode)), offset
    # (index of the first message shown in UIs), sep_style, sep/sep2 separators, version tag, optional
    # tokenizer (needed for LLAMA_3), stop_str/stop_token_ids generation stop criteria and skip_next flag.
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        # [FUNCTION DOC] Renders the messages into a prompt string. If the first message is a tuple its text
        # is normalized: for
        # "mmtag" versions "<image>" is removed and two synthetic turns ("<Image><image></Image>",
        # "Received.") are
        # prepended; otherwise "<image>" is moved to the start followed by a newline.
        # Note: in that case the first message is replaced by the bare text, losing the tuple.
        # Styles: SINGLE "system sep role: msg sep ..."; TWO alternates sep/sep2; CHATML "system sep\n" then
        # "role\nmsg sep\n" with "<image>" repeated per image for tuple messages (returns immediately);
        # LLAMA_3 uses
        # tokenizer.apply_chat_template with a system message and generation prompt (ValueError if no
        # tokenizer);
        # MPT "role msg sep"; GEMMA like MPT but asserts roles alternate; LLAMA_2 wraps system in <<SYS>> and
        # user
        # turns in [INST]...[/INST] with sep2 after assistant turns; PLAIN concatenates messages with
        # alternating
        # sep/sep2. Empty messages render just the role prefix. Unknown style raises ValueError.
        # KNOWN ISSUE: PLAIN (conv_plain, the default) has sep2 None, so a second (assistant) message raises
        # TypeError when concatenating.
        # KNOWN ISSUE: LLAMA_3 unpacks tuple messages into 2 items while every other branch expects 3-tuples.
        # KNOWN ISSUE: LLAMA_2 uses lstrip(self.sep), which strips a character set, not the separator prefix.
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            if self.tokenizer is None:
                raise ValueError("Llama 3 tokenizer is not available. Make sure you have the necessary permissions.")
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})

            # print(chat_template_messages)
            return self.tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True)
            # ret = "" if self.system == "" else self.system + self.sep + "\n"
            # for role, message in messages:
            #     if message:
            #         if type(message) is tuple:
            #             message, images = message
            #             message = "<image>" * len(images) + message
            #         ret += role + "\n" + message + self.sep + "\n"
            #     else:
            #         ret += role + "\n"
            # return ret

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = ""
            for i, (role, message) in enumerate(messages):
                assert role == self.roles[i % 2], "Conversation should alternate user/assistant/user/assistant/..."
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        # [FUNCTION DOC] Appends [role, message] to the message list (mutates self.messages).
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format="PNG"):
        # [FUNCTION DOC] Prepares one image for display: "Pad" pads to a square with expand2square,
        # "Default"/"Crop" leave it,
        # "Resize" resizes to 336x336, other modes raise ValueError. Non-PIL inputs (paths/file objects) are
        # opened
        # and converted to RGB. Then resizes so the shortest edge is min(672/aspect, 448, current short side)
        # and
        # the long edge keeps the aspect ratio. Returns the PIL image if return_pil, else a base64 string of
        # the
        # image encoded as image_format (default PNG).
        # Note: padding/resizing modes are applied before a path is opened, so they only work for PIL inputs.
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
                # [FUNCTION DOC] HELPER: pads a PIL image to a square canvas of its longest side filled with
                # background_color (default
                # (122, 116, 104)), centering the original; returns square images unchanged.
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if type(image) is not Image.Image:
            image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False, return_path=False):
        # [FUNCTION DOC] Collects images from user turns (even positions after offset) whose message is a
        # tuple (text, image or
        # list of images, mode). Returns a list.
        # KNOWN ISSUE: when not return_path and the item is an image file, the processed image is computed but
        # never
        # appended, so only non-image items (or everything when return_path=True) are returned.
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        if not return_path and self.is_image_file(img):
                            img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        else:
                            images.append(img)
        return images

    def is_image_file(self, filename):
        # [FUNCTION DOC] Returns True if filename ends (case-insensitively) with a common image extension.
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def is_video_file(self, filename):
        # [FUNCTION DOC] Returns True if filename ends (case-insensitively) with a common video extension.
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"]
        return any(filename.lower().endswith(ext) for ext in video_extensions)

    def to_gradio_chatbot(self):
        # [FUNCTION DOC] Converts messages after offset to Gradio chatbot pairs [user_msg, bot_msg]. For tuple
        # user turns, normalizes
        # "<image>" placement, embeds image files as inline base64 JPEG <img> HTML tags prepended to the text,
        # and
        # adds video files as separate ((path,), None) entries. Assistant turns fill the second slot of the
        # last pair.
        # Returns the list.
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    if len(image) == 1:
                        msg = "<image>\n" + msg.replace("<image>", "").strip()
                    else:
                        msg = re.sub(r"(<image>)\n(?=<image>)", r"\1 ", msg)

                    img_str_list = []                         
                    for img in image:
                        if self.is_image_file(img):
                            img_b64_str = self.process_image(img, "Default", return_pil=False, image_format="JPEG")
                            img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" style="max-width: 256px; max-height: 256px; width: auto; height: auto; object-fit: contain;"/>'
                            img_str_list.append(img_str)
                        elif self.is_video_file(img):
                            ret.append(((img,), None))

                    msg = msg.strip()
                    img_place_holder = ""
                    for img_str in img_str_list:
                        img_place_holder += f"{img_str}\n\n"

                    if len(img_str_list) > 0:
                        msg = f"{img_place_holder}\n\n{msg}"

                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        # [FUNCTION DOC] Returns a new Conversation with a copied message list (inner pairs copied) and the
        # same system, roles,
        # offset, separators and version. tokenizer, stop criteria and skip_next are not copied (reset to
        # defaults).
        return Conversation(system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages], offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version)

    def dict(self):
        # [FUNCTION DOC] Returns a JSON-friendly dict of system, roles, messages, offset, sep, sep2; when
        # get_images() is non-empty
        # tuple messages are reduced to their text.
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

def safe_load_tokenizer(tokenizer_id):
    # [FUNCTION DOC] Returns AutoTokenizer.from_pretrained(tokenizer_id), or None if loading raises any
    # exception.
    try:
        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        return None


conv_qwen = Conversation(
    system="""<|im_start|>system
You are a helpful assistant.""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)


conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_plain
conv_templates = {
    "default": conv_qwen,
    "plain": conv_plain,
    "chatml_direct": conv_chatml_direct,
    "qwen_1_5": conv_qwen,
    "qwen_2": conv_qwen,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())