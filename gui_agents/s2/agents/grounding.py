import ast
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import pytesseract
from PIL import Image
from pytesseract import Output

from gui_agents.s2.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2.core.mllm import LMMAgent
from gui_agents.s2.utils.common_utils import (
    call_llm_safe,
    parse_single_code_from_string,
)


# 기본 ACI(Agent-Computer Interface) 클래스
class ACI:
    def __init__(self):
        # 지식 저장소 - save_to_knowledge 메서드로 저장된 정보를 보관
        self.notes: List[str] = []


# 에이전트 액션 데코레이터: 메서드를 에이전트의 행동으로 표시
def agent_action(func):
    func.is_agent_action = True
    return func


UBUNTU_APP_SETUP = f"""import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""


SET_CELL_VALUES_CMD = """import uno
import subprocess

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"

    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"

    if component.supportsService("com.sun.star.sheet.PresentationDocument"):
        return "Impress"

    return None

def cell_ref_to_indices(cell_ref):
    column_letters = ''.join(filter(str.isalpha, cell_ref))
    row_number = ''.join(filter(str.isdigit, cell_ref))

    col = sum((ord(char.upper()) - ord('A') + 1) * (26**idx) for idx, char in enumerate(reversed(column_letters))) - 1
    row = int(row_number) - 1
    return col, row

def set_cell_values(new_cell_values: dict[str, str], app_name: str = "Untitled 1", sheet_name: str = "Sheet1"):
    new_cell_values_idx = {{}}
    for k, v in new_cell_values.items():
        try:
            col, row = cell_ref_to_indices(k)
        except:
            col = row = None

        if col is not None and row is not None:
            new_cell_values_idx[(col, row)] = v

    # Clean up previous TCP connections.
    subprocess.run(
        'echo \"password\" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        check=True,
        text=True,
        capture_output=True
    )

    # Dynamically allow soffice to listen on port 2002.
    subprocess.run(
        [
            "soffice",
            "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"
        ]
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    context = resolver.resolve(
        f"uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Collect all LibreOffice-related opened windows.
    documents = []
    for i, component in enumerate(desktop.Components):
        title = component.Title
        doc_type = identify_document_type(component)
        documents.append((i, component, title, doc_type))

    # Find the LibreOffice Calc app and the sheet of interest.
    spreadsheet = [doc for doc in documents if doc[3] == "Calc"]
    selected_spreadsheet = [doc for doc in spreadsheet if doc[2] == app_name]
    if spreadsheet:
        try:
            if selected_spreadsheet:
                spreadsheet = selected_spreadsheet[0][1]
            else:
                spreadsheet = spreadsheet[0][1]

            sheet = spreadsheet.Sheets.getByName(sheet_name)
        except:
            raise ValueError(f"Could not find sheet {{sheet_name}} in {{app_name}}.")

        for (col, row), value in new_cell_values_idx.items():
            cell = sheet.getCellByPosition(col, row)

            # Set the cell value.
            if isinstance(value, (int, float)):
                cell.Value = value
            elif isinstance(value, str):
                if value.startswith("="):
                    cell.Formula = value
                else:
                    cell.String = value
            elif isinstance(value, bool):
                cell.Value = 1 if value else 0
            elif value is None:
                cell.clearContents(0)
            else:
                raise ValueError(f"Unsupported cell value type: {{type(value)}}")

    else:
        raise ValueError(f"Could not find LibreOffice Calc app corresponding to {{app_name}}.")

set_cell_values(new_cell_values={cell_values}, app_name="{app_name}", sheet_name="{sheet_name}")        
"""


# OSWorldACI 클래스: 실제 OS와 상호작용하는 구현체
class OSWorldACI(ACI):
    def __init__(
        self,
        platform: str,                      # 운영체제 플랫폼 (macos, ubuntu, windows)
        engine_params_for_generation: Dict,  # 텍스트 생성용 LLM 설정
        engine_params_for_grounding: Dict,   # 이미지 인식용 LLM 설정
        width: int = 1920,                  # 화면 너비
        height: int = 1080,                 # 화면 높이
    ):
        # 플랫폼 정보 저장 (OS별 다른 동작 수행)
        self.platform = platform
        
        # 화면 좌표 스케일링 정보
        self.width = width
        self.height = height
        
        # 에이전트 지식 저장소
        self.notes = []
        
        # 액션 실행에 사용될 좌표 정보
        self.coords1 = None
        self.coords2 = None
        
        # UI 요소 인식을 위한 멀티모달 LLM 초기화
        self.grounding_model = LMMAgent(engine_params_for_grounding)
        self.engine_params_for_grounding = engine_params_for_grounding
        
        # 텍스트 인식을 위한 LLM 초기화
        self.text_span_agent = LMMAgent(
            engine_params=engine_params_for_generation,
            system_prompt=PROCEDURAL_MEMORY.PHRASE_TO_WORD_COORDS_PROMPT,
        )

    # 참조 표현식(설명)을 기반으로 UI 요소의 좌표 생성
    def generate_coords(self, ref_expr: str, obs: Dict) -> List[int]:
        # 그라운딩 모델 상태 초기화
        self.grounding_model.reset()
        
        # 프롬프트 구성: 사용자 설명을 좌표로 변환하도록 요청
        prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point in your response.\n"
        
        # 멀티모달 LLM에 스크린샷과 함께 쿼리 전달
        self.grounding_model.add_message(
            text_content=prompt, image_content=obs["screenshot"], put_text_last=True
        )
        
        # 좌표 생성 및 파싱
        response = call_llm_safe(self.grounding_model)
        print("RAW GROUNDING MODEL RESPONSE:", response)
        numericals = re.findall(r"\d+", response)
        assert len(numericals) >= 2
        return [int(numericals[0]), int(numericals[1])]

    # OCR을 사용하여 스크린샷에서 텍스트 요소와 위치 추출
    def get_ocr_elements(self, b64_image_data: str) -> Tuple[str, List]:
        # 이미지 로드 및 OCR 처리
        image = Image.open(BytesIO(b64_image_data))
        image_data = pytesseract.image_to_data(image, output_type=Output.DICT)
        
        # 텍스트 정리 및 구조화
        for i, word in enumerate(image_data["text"]):
            image_data["text"][i] = re.sub(
                r"^[^a-zA-Z\s.,!?;:\-\+]+|[^a-zA-Z\s.,!?;:\-\+]+$", "", word
            )
        
        # 인식된 각 단어에 대한 정보 추출 (위치, 크기 등)
        ocr_elements = []
        ocr_table = "Text Table:\nWord id\tText\n"
        grouping_map = defaultdict(list)
        ocr_id = 0
        
        # 텍스트 블록 단위로 정보 구성
        for i in range(len(image_data["text"])):
            block_num = image_data["block_num"][i]
            if image_data["text"][i]:
                grouping_map[block_num].append(image_data["text"][i])
                ocr_table += f"{ocr_id}\t{image_data['text'][i]}\n"
                ocr_elements.append({
                    "id": ocr_id,
                    "text": image_data["text"][i],
                    "group_num": block_num,
                    "word_num": len(grouping_map[block_num]),
                    "left": image_data["left"][i],
                    "top": image_data["top"][i],
                    "width": image_data["width"][i],
                    "height": image_data["height"][i],
                })
                ocr_id += 1
        
        return ocr_table, ocr_elements

    # Given the state and worker's text phrase, generate the coords of the first/last word in the phrase
    def generate_text_coords(
        self, phrase: str, obs: Dict, alignment: str = ""
    ) -> List[int]:

        ocr_table, ocr_elements = self.get_ocr_elements(obs["screenshot"])

        alignment_prompt = ""
        if (alignment == "start"):
            alignment_prompt = "**Important**: Output the word id of the FIRST word in the provided phrase.\n"
        elif (alignment == "end"):
            alignment_prompt = "**Important**: Output the word id of the LAST word in the provided phrase.\n"

        # Load LLM prompt
        self.text_span_agent.reset()
        self.text_span_agent.add_message(
            alignment_prompt + "Phrase: " + phrase + "\n" + ocr_table, role="user"
        )
        self.text_span_agent.add_message(
            "Screenshot:\n", image_content=obs["screenshot"], role="user"
        )

        # Obtain the target element
        response = call_llm_safe(self.text_span_agent)
        print("TEXT SPAN AGENT RESPONSE:", response)
        numericals = re.findall(r"\d+", response)
        if len(numericals) > 0:
            text_id = int(numericals[-1])
        else:
            text_id = 0
        elem = ocr_elements[text_id]

        # Compute the element coordinates
        if alignment == "start":
            coords = [elem["left"], elem["top"] + (elem["height"] // 2)]
        elif alignment == "end":
            coords = [elem["left"] + elem["width"], elem["top"] + (elem["height"] // 2)]
        else:
            coords = [
                elem["left"] + (elem["width"] // 2),
                elem["top"] + (elem["height"] // 2),
            ]
        return coords

    # Takes a description based action and assigns the coordinates for any coordinate based action
    # Raises an error if function can't be parsed
   # 에이전트 행동에 필요한 좌표 할당
    def assign_coordinates(self, plan: str, obs: Dict):
        # 이전 액션의 좌표 초기화
        self.coords1, self.coords2 = None, None
        
        try:
            # 액션 파싱: 함수명과 인자 추출
            action = parse_single_code_from_string(plan.split("Grounded Action")[-1])
            function_name = re.match(r"(\w+\.\w+)\(", action).group(1)
            args = self.parse_function_args(action)
        except Exception as e:
            raise RuntimeError(f"Error in parsing grounded action: {e}") from e
        
        # 액션 유형에 따른 좌표 생성
        # 1. 단일 좌표가 필요한 액션 (클릭, 입력, 스크롤)
        if (
            function_name in ["agent.click", "agent.type", "agent.scroll"]
            and len(args) >= 1
            and args[0] != None
        ):
            self.coords1 = self.generate_coords(args[0], obs)
        # 2. 두 개의 좌표가 필요한 액션 (드래그 앤 드롭)
        elif function_name == "agent.drag_and_drop" and len(args) >= 2:
            self.coords1 = self.generate_coords(args[0], obs)
            self.coords2 = self.generate_coords(args[1], obs)
        # 3. 텍스트 범위를 위한 특수 좌표 할당
        elif function_name == "agent.highlight_text_span" and len(args) >= 2:
            self.coords1 = self.generate_text_coords(args[0], obs, alignment="start")
            self.coords2 = self.generate_text_coords(args[1], obs, alignment="end")

    # Resize from grounding model dim into OSWorld dim (1920 * 1080)
    def resize_coordinates(self, coordinates: List[int]) -> List[int]:
        # User explicitly passes the grounding model dimensions
        if {"grounding_width", "grounding_height"}.issubset(
            self.engine_params_for_grounding
        ):
            grounding_width = self.engine_params_for_grounding["grounding_width"]
            grounding_height = self.engine_params_for_grounding["grounding_height"]
        # Default to (1000, 1000), which is UI-TARS resizing
        else:
            grounding_width = 1000
            grounding_height = 1000

        return [
            round(coordinates[0] * self.width / grounding_width),
            round(coordinates[1] * self.height / grounding_height),
        ]

    # Given a generated ACI function, returns a list of argument values, where descriptions are at the front of the list
    def parse_function_args(self, function: str) -> List[str]:
        tree = ast.parse(function)
        call_node = tree.body[0].value

        def safe_eval(node):
            if isinstance(
                node, ast.Constant
            ):  # Handles literals like numbers, strings, etc.
                return node.value
            else:
                return ast.unparse(node)  # Return as a string if not a literal

        positional_args = [safe_eval(arg) for arg in call_node.args]
        keyword_args = {kw.arg: safe_eval(kw.value) for kw in call_node.keywords}

        res = []

        for key, val in keyword_args.items():
            if "description" in key:
                res.append(val)

        for arg in positional_args:
            res.append(arg)

        return res

    @agent_action
    def click(
        self,
        element_description: str,       # 클릭할 요소 설명
        num_clicks: int = 1,            # 클릭 횟수
        button_type: str = "left",      # 마우스 버튼 타입 (left, right, middle)
        hold_keys: List = [],           # 동시에 누를 키 목록
    ):
        """
        설명에 기반하여 화면 요소 클릭
        - element_description: 클릭할 대상에 대한 자세한 설명
        - num_clicks: 클릭 횟수 설정
        - button_type: 마우스 버튼 종류 
        - hold_keys: 클릭하는 동안 함께 누를 키 (단축키 등)
        """
        # 모델이 생성한 원시 좌표를 실제 화면 좌표로 스케일링
        x, y = self.resize_coordinates(self.coords1)
        
        # pyautogui 명령어 구성
        command = "import pyautogui; "
        
        # 키 누르기 명령 추가
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        
        # 클릭 명령 추가
        command += f"""import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); """
        
        # 키 떼기 명령 추가
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
            
        return command  # 실행할 Python 코드 반환

    @agent_action
    def switch_applications(self, app_code):
        """Switch to a different application that is already open
        Args:
            app_code:str the code name of the application to switch to from the provided list of open applications
        """
        if self.platform == "mac":
            return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"
        elif self.platform == "ubuntu":
            return UBUNTU_APP_SETUP.replace("APP_NAME", app_code)
        elif self.platform == "windows":
            return f"import pyautogui; import time; pyautogui.hotkey('win', 'd', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"

    @agent_action
    def open(self, app_or_filename: str):
        """Open any application or file with name app_or_filename. Use this action to open applications or files on the desktop, do not open manually.
        Args:
            app_or_filename:str, the name of the application or filename to open
        """
        return f"import pyautogui; pyautogui.hotkey('win'); time.sleep(0.5); pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); pyautogui.hotkey('enter'); time.sleep(0.5)"

    @agent_action
    def type(
        self,
        element_description: Optional[str] = None,  # 입력할 요소 설명 (없으면 현재 위치)
        text: str = "",                            # 입력할 텍스트
        overwrite: bool = False,                   # 기존 텍스트 덮어쓰기 여부
        enter: bool = False,                       # 입력 후 엔터 키 누름 여부
    ):
        """텍스트 입력 액션"""
        # 타겟 요소 있으면 해당 위치로 이동
        if self.coords1 is not None:
            x, y = self.resize_coordinates(self.coords1)
            command = "import pyautogui; "
            command += f"pyautogui.click({x}, {y}); "
            
            # 덮어쓰기 옵션이면 전체 선택 후 삭제
            if overwrite:
                command += f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
            
            # 텍스트 입력
            command += f"pyautogui.write({repr(text)}); "
            
            # 엔터 키 옵션
            if enter:
                command += "pyautogui.press('enter'); "
        else:
            # 타겟 없이 현재 커서 위치에서 입력
            command = "import pyautogui; "
            
            if overwrite:
                command += f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
            
            command += f"pyautogui.write({repr(text)}); "
            
            if enter:
                command += "pyautogui.press('enter'); "
        
        return command

    @agent_action
    def save_to_knowledge(self, text: List[str]):
        """Save facts, elements, texts, etc. to a long-term knowledge bank for reuse during this task. Can be used for copy-pasting text, saving elements, etc.
        Args:
            text:List[str] the text to save to the knowledge
        """
        self.notes.extend(text)
        return """WAIT"""

    @agent_action
    def drag_and_drop(
        self, starting_description: str, ending_description: str, hold_keys: List = []
    ):
        """Drag from the starting description to the ending description
        Args:
            starting_description:str, a very detailed description of where to start the drag action. This description should be at least a full sentence.
            ending_description:str, a very detailed description of where to end the drag action. This description should be at least a full sentence.
            hold_keys:List list of keys to hold while dragging
        """
        x1, y1 = self.resize_coordinates(self.coords1)
        x2, y2 = self.resize_coordinates(self.coords2)

        command = "import pyautogui; "

        command += f"pyautogui.moveTo({x1}, {y1}); "
        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        # Return pyautoguicode to drag and drop the elements

        return command

    @agent_action
    def highlight_text_span(self, starting_phrase: str, ending_phrase: str):
        """Highlight a text span between a provided starting phrase and ending phrase. Use this to highlight words, lines, and paragraphs.
        Args:
            starting_phrase:str, the phrase that denotes the start of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
            ending_phrase:str, the phrase that denotes the end of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
        """

        x1, y1 = self.coords1
        x2, y2 = self.coords2

        command = "import pyautogui; "
        command += f"pyautogui.moveTo({x1}, {y1}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "

        # Return pyautoguicode to drag and drop the elements
        return command

    @agent_action
    def set_cell_values(
        self, cell_values: Dict[str, Any], app_name: str, sheet_name: str
    ):
        """Use this to set individual cell values in a spreadsheet. For example, setting A2 to "hello" would be done by passing {"A2": "hello"} as cell_values. The sheet must be opened before this command can be used.
        Args:
            cell_values: Dict[str, Any], A dictionary of cell values to set in the spreadsheet. The keys are the cell coordinates in the format "A1", "B2", etc.
                Supported value types include: float, int, string, bool, formulas.
            app_name: str, The name of the spreadsheet application.
            sheet_name: str, The name of the sheet in the spreadsheet.
        """
        return SET_CELL_VALUES_CMD.format(
        )

    @agent_action
    def scroll(self, element_description: str, clicks: int, shift: bool = False):
        """Scroll the element in the specified direction
        Args:
            element_description:str, a very detailed description of which element to enter scroll in. This description should be at least a full sentence.
            clicks:int, the number of clicks to scroll can be positive (up) or negative (down).
            shift:bool, whether to use shift+scroll for horizontal scrolling
        """

        x, y = self.resize_coordinates(self.coords1)

        if shift:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.hscroll({clicks})"
        else:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.vscroll({clicks})"

    @agent_action
    def hotkey(self, keys: List):
        """Press a hotkey combination
        Args:
            keys:List the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
        """
        # add quotes around the keys
        keys = [f"'{key}'" for key in keys]
        return f"import pyautogui; pyautogui.hotkey({', '.join(keys)})"

    @agent_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold a list of keys and press a list of keys
        Args:
            hold_keys:List, list of keys to hold
            press_keys:List, list of keys to press in a sequence
        """

        press_keys_str = "[" + ", ".join([f"'{key}'" for key in press_keys]) + "]"
        command = "import pyautogui; "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.press({press_keys_str}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        return command

    @agent_action
    def wait(self, time: float):
        """Wait for a specified amount of time
        Args:
            time:float the amount of time to wait in seconds
        """
        return f"""import time; time.sleep({time})"""

    @agent_action
    def done(
        self,
        return_value: Optional[Union[Dict, str, List, Tuple, int, float, bool]] = None,
    ):
        """End the current task with a success and the required return value"""
        self.returned_info = return_value
        return """DONE"""

    @agent_action
    def fail(self):
        """End the current task with a failure, and replan the whole task."""
        return """FAIL"""
