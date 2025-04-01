import pyautogui

# 1. 현재 화면 크기 출력
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# 2. 마우스 위치 출력
current_mouse_x, current_mouse_y = pyautogui.position()
print(f"Current mouse position: {current_mouse_x}, {current_mouse_y}")

# 3. 마우스 이동 테스트
print("Moving mouse to (100, 100)...")
pyautogui.moveTo(100, 100, duration=1)

# 4. 키보드 입력 테스트
print("Typing 'Hello, PyAutoGUI!'...")
pyautogui.typewrite("Hello, PyAutoGUI!", interval=0.1)

# 5. 마우스 클릭 테스트
print("Clicking at (100, 100)...")
pyautogui.click(100, 100)