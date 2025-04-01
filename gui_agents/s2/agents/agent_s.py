import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

# ACI(Agent-Computer Interface)는 컴퓨터와 상호작용하는 인터페이스를 제공
from gui_agents.s2.agents.grounding import ACI
# Worker는 실제 작업 실행을 담당하는 컴포넌트
from gui_agents.s2.agents.worker import Worker
# Manager는 작업 계획 및 관리를 담당하는 컴포넌트
from gui_agents.s2.agents.manager import Manager
# DAG(Directed Acyclic Graph) 구조를 위한 Node 클래스
from gui_agents.s2.utils.common_utils import Node

# 로깅 설정
logger = logging.getLogger("desktopenv.agent")
# 현재 파일의 디렉토리 경로
working_dir = os.path.dirname(os.path.abspath(__file__))


class UIAgent:
    """
    UI 자동화 에이전트의 기본 클래스
    
    이 클래스는 모든 UI 에이전트의 공통 기능과 인터페이스를 정의합니다.
    실제 구현은 하위 클래스(예: GraphSearchAgent)에서 이루어집니다.
    """

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "macos",
        action_space: str = "pyautogui",
        observation_type: str = "a11y_tree",
        search_engine: str = "perplexica",
    ):
        """
        UIAgent 초기화 함수
        
        Args:
            engine_params: LLM 엔진 구성 매개변수 (API 키, 모델 등)
            grounding_agent: UI 상호작용을 위한 ACI 클래스 인스턴스
            platform: 운영체제 플랫폼 (macos, linux, windows)
            action_space: 사용할 액션 공간 유형 (pyautogui, aci)
            observation_type: 사용할 관찰 유형 (a11y_tree, mixed)
            search_engine: 사용할 검색 엔진 (perplexica, LLM)
        """
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.engine = search_engine

    # 에이전트 상태 초기화
    def reset(self) -> None:
        """Reset agent state"""
        pass

    # 다음 행동 예측
    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction

        Args:
            instruction: Natural language instruction
            observation: Current UI state observation

        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        pass

    # 전체 작업의 서술적 메모리 업데이트
    def update_narrative_memory(self, trajectory: str) -> None:
        """Update narrative memory with task trajectory

        Args:
            trajectory: String containing task execution trajectory
        """
        pass

    # 서브태스크별 경험 기록
    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        """Update episodic memory with subtask trajectory

        Args:
            meta_data: Metadata about current subtask execution
            subtask_trajectory: String containing subtask execution trajectory

        Returns:
            Updated subtask trajectory
        """
        pass


class GraphSearchAgent(UIAgent):
    """Agent that uses hierarchical planning and directed acyclic graph modeling for UI automation"""
    """
    계층적 계획과 방향성 비순환 그래프(DAG)를 사용하는 UI 자동화 에이전트
    
    이 클래스는 복잡한 작업을 여러 하위 작업으로 분할하고, 그래프 구조로 관리하여
    효율적으로 실행하는 핵심 로직을 구현합니다.
    """

    def __init__(
        # UIAgent 초기화
        # memory_root_path: 메모리 저장 루트 경로
        # memory_folder_name: 메모리 폴더명 (기본값: "kb")
        
        # 지식 베이스 초기화: 이전에 학습한 내용이나 기본 지식을 로드
        # local_kb_path: 플랫폼별 지식 베이스 경로
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "macos",
        action_space: str = "pyautogui",
        observation_type: str = "mixed",
        search_engine: Optional[str] = None,
        memory_root_path: str = os.getcwd(),
        memory_folder_name: str = "kb",
    ):
        """Initialize GraphSearchAgent

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, ubuntu)
            action_space: Type of action space to use (pyautogui, other)
            observation_type: Type of observations to use (a11y_tree, screenshot, mixed)
            search_engine: Search engine to use (LLM, perplexica)
            memory_root_path: Path to memory directory. Defaults to current working directory.
            memory_folder_name: Name of memory folder. Defaults to "kb".
        """
        super().__init__(
            engine_params,
            grounding_agent,
            platform,
            action_space,
            observation_type,
            search_engine,
        )

        self.memory_root_path = memory_root_path
        self.memory_folder_name = memory_folder_name

        # Initialize agent's knowledge base on user's current working directory.
        print("Downloading knowledge base initial Agent-S knowledge...")
        self.local_kb_path = os.path.join(
            self.memory_root_path, self.memory_folder_name, self.platform
        )

        library_kb_path = os.path.join(working_dir, "../kb", self.platform)
        if not os.path.exists(self.local_kb_path):
            shutil.copytree(library_kb_path, self.local_kb_path)
            print("Successfully completed download of knowledge base.")
        else:
            print(
                f"Path local_kb_path {self.local_kb_path} already exists. Skipping download."
            )
            print(
                f"If you'd like to re-download the initial knowledge base, please delete the existing knowledge base at {self.local_kb_path}."
            )

        self.reset()

    def reset(self) -> None:
        # 계획자(Planner)와 실행자(Worker) 초기화
        # self.planner: 전체 작업을 서브태스크로 분할하는 컴포넌트
        # self.executor: 개별 서브태스크를 실행하는 컴포넌트
        
        # 상태 변수 초기화: 작업 추적을 위한 변수들
        # requires_replan: 재계획 필요 여부
        # needs_next_subtask: 다음 서브태스크 필요 여부
        # step_count, turn_count: 실행 단계 추적
        # completed_tasks, subtasks: 완료된/남은 작업 목록
        # current_subtask: 현재 진행 중인 서브태스크
        # search_query: 현재 작업 검색 쿼리
        """Reset agent state and initialize components"""
        # Initialize core components
        self.planner = Manager(
            self.engine_params,
            self.grounding_agent,
            platform=self.platform,
            search_engine=self.engine,
            local_kb_path=self.local_kb_path,
        )
        self.executor = Worker(
            self.engine_params,
            self.grounding_agent,
            platform=self.platform,
            use_subtask_experience=True,
            local_kb_path=self.local_kb_path,
        )

        # Reset state variables
        self.requires_replan: bool = True
        self.needs_next_subtask: bool = True
        self.step_count: int = 0
        self.turn_count: int = 0
        self.failure_subtask: Optional[Node] = None
        self.should_send_action: bool = False
        self.completed_tasks: List[Node] = []
        self.current_subtask: Optional[Node] = None
        self.subtasks: List[Node] = []
        self.search_query: str = ""
        self.subtask_status: str = "Start"

    # 실행자 상태와 단계 카운터 초기화
    def reset_executor_state(self) -> None:
        """Reset executor and step counter"""
        self.executor.reset()
        self.step_count = 0

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        # 핵심 의사 결정 로직:
        # 1. 필요시 작업을 서브태스크로 분할하는 계획 생성
        # 2. 다음 수행할 서브태스크 선택
        # 3. 선택된 서브태스크에 대한 다음 행동 생성
        # 4. 실패/완료 처리 및 상태 업데이트
        # 5. 결과 및 행동 반환
        
        # Initialize the three info dictionaries
        planner_info = {}
        executor_info = {}
        evaluator_info = {
            "obs_evaluator_response": "",
            "num_input_tokens_evaluator": 0,
            "num_output_tokens_evaluator": 0,
            "evaluator_cost": 0.0,
        }
        actions = []

        # If the DONE response by the executor is for a subtask, then the agent should continue with the next subtask without sending the action to the environment
        while not self.should_send_action:
            self.subtask_status = "In"
            # If replan is true, generate a new plan. True at start, after a failed plan, or after subtask completion
            if self.requires_replan:
                logger.info("(RE)PLANNING...")
                planner_info, self.subtasks = self.planner.get_action_queue(
                    instruction=instruction,
                    observation=observation,
                    failed_subtask=self.failure_subtask,
                    completed_subtasks_list=self.completed_tasks,
                    remaining_subtasks_list=self.subtasks,
                )

                self.requires_replan = False
                if "search_query" in planner_info:
                    self.search_query = planner_info["search_query"]
                else:
                    self.search_query = ""

            # use the exectuor to complete the topmost subtask
            if self.needs_next_subtask:
                logger.info("GETTING NEXT SUBTASK...")

                # this can be empty if the DAG planner deems that all subtasks are completed
                if len(self.subtasks) <= 0:
                    self.requires_replan = True
                    self.needs_next_subtask = True
                    self.failure_subtask = None
                    self.completed_tasks.append(self.current_subtask)

                    # reset executor state
                    self.reset_executor_state()
                    self.should_send_action = True
                    self.subtask_status = "Done"
                    executor_info = {
                        "executor_plan": "agent.done()",
                        "plan_code": "agent.done()",
                        "reflection": "agent.done()",
                    }
                    actions = ["DONE"]
                    break

                self.current_subtask = self.subtasks.pop(0)
                logger.info(f"NEXT SUBTASK: {self.current_subtask}")
                self.needs_next_subtask = False
                self.subtask_status = "Start"

            # get the next action from the executor
            executor_info, actions = self.executor.generate_next_action(
                instruction=instruction,
                search_query=self.search_query,
                subtask=self.current_subtask.name,
                subtask_info=self.current_subtask.info,
                future_tasks=self.subtasks,
                done_task=self.completed_tasks,
                obs=observation,
            )

            self.step_count += 1

            # set the should_send_action flag to True if the executor returns an action
            self.should_send_action = True

            # replan on failure
            if "FAIL" in actions:
                self.requires_replan = True
                self.needs_next_subtask = True

                # assign the failed subtask
                self.failure_subtask = self.current_subtask

                # reset the step count, executor, and evaluator
                self.reset_executor_state()

                # if more subtasks are remaining, we don't want to send DONE to the environment but move on to the next subtask
                if self.subtasks:
                    self.should_send_action = False

            # replan on subtask completion
            elif "DONE" in actions:
                self.requires_replan = True
                self.needs_next_subtask = True
                self.failure_subtask = None
                self.completed_tasks.append(self.current_subtask)

                # reset the step count, executor, and evaluator
                self.reset_executor_state()

                # if more subtasks are remaining, we don't want to send DONE to the environment but move on to the next subtask
                if self.subtasks:
                    self.should_send_action = False
                self.subtask_status = "Done"

            self.turn_count += 1

        # reset the should_send_action flag for next iteration
        self.should_send_action = False

        # concatenate the three info dictionaries
        info = {
            **{
                k: v
                for d in [planner_info or {}, executor_info or {}, evaluator_info or {}]
                for k, v in d.items()
            }
        }
        info.update(
            {
                "subtask": self.current_subtask.name,
                "subtask_info": self.current_subtask.info,
                "subtask_status": self.subtask_status,
            }
        )

        return info, actions

    def update_narrative_memory(self, trajectory: str) -> None:
        # 전체 작업 수행 과정을 요약하여 narrative_memory.json에 저장
        # trajectory: 작업 수행 과정이 담긴 문자열
        """Update narrative memory from task trajectory

        Args:
            trajectory: String containing task execution trajectory
        """
        try:
            reflection_path = os.path.join(self.local_kb_path, "narrative_memory.json")
            try:
                reflections = json.load(open(reflection_path))
            except:
                reflections = {}

            if self.search_query not in reflections:
                reflection = self.planner.summarize_narrative(trajectory)
                reflections[self.search_query] = reflection

            with open(reflection_path, "w") as f:
                json.dump(reflections, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update narrative memory: {e}")

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        # 개별 서브태스크 수행 경험을 episodic_memory.json에 저장
        # 서브태스크 시작/진행/완료에 따른 메모리 업데이트 처리
        """Update episodic memory from subtask trajectory

        Args:
            meta_data: Metadata about current subtask execution
            subtask_trajectory: String containing subtask execution trajectory

        Returns:
            Updated subtask trajectory
        """
        subtask = meta_data["subtask"]
        subtask_info = meta_data["subtask_info"]
        subtask_status = meta_data["subtask_status"]
        # Handle subtask trajectory
        if subtask_status == "Start" or subtask_status == "Done":
            # If it's a new subtask start, finalize the previous subtask trajectory if it exists
            if subtask_trajectory:
                subtask_trajectory += "\nSubtask Completed.\n"
                subtask_key = subtask_trajectory.split(
                    "\n----------------------\n\nPlan:\n"
                )[0]
                try:
                    subtask_path = os.path.join(
                        self.local_kb_path, "episodic_memory.json"
                    )
                    kb = json.load(open(subtask_path))
                except:
                    kb = {}
                if subtask_key not in kb.keys():
                    subtask_summarization = self.planner.summarize_episode(
                        subtask_trajectory
                    )
                    kb[subtask_key] = subtask_summarization
                else:
                    subtask_summarization = kb[subtask_key]
                logger.info("subtask_key: %s", subtask_key)
                logger.info("subtask_summarization: %s", subtask_summarization)
                with open(subtask_path, "w") as fout:
                    json.dump(kb, fout, indent=2)
                # Reset for the next subtask
                subtask_trajectory = ""
            # Start a new subtask trajectory
            subtask_trajectory = (
                "Task:\n"
                + self.search_query
                + "\n\nSubtask: "
                + subtask
                + "\nSubtask Instruction: "
                + subtask_info
                + "\n----------------------\n\nPlan:\n"
                + meta_data["executor_plan"]
                + "\n"
            )
        elif subtask_status == "In":
            # Continue appending to the current subtask trajectory if it's still ongoing
            subtask_trajectory += (
                "\n----------------------\n\nPlan:\n"
                + meta_data["executor_plan"]
                + "\n"
            )

        return subtask_trajectory
