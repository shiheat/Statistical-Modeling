# -*- coding: utf-8 -*-

import random
from typing import List, Dict, Optional, Tuple, Any

# --- Part 1: Element Database ---
# Represents the storage and retrieval of design elements.

class ElementDatabase:
    """Manages the collection of design elements."""

    def __init__(self, initial_data: Optional[List[Dict[str, Any]]] = None):
        """Initializes the database."""
        self._elements: List[Dict[str, Any]] = []
        if initial_data:
            self.load_data(initial_data)
        print(f"数据库初始化完成，包含 {len(self._elements)} 个元素。")

    def load_data(self, data: List[Dict[str, Any]]):
        """Loads element data into the database."""
        # In a real system, this might connect to a SQL/NoSQL DB and load.
        self._elements = data
        print(f"成功加载 {len(data)} 个元素到数据库。")

    def get_all_elements(self) -> List[Dict[str, Any]]:
        """Returns all elements in the database."""
        return self._elements

    def get_elements_by_style(self, style: str) -> List[Dict[str, Any]]:
        """
        Retrieves elements filtered by a specific style (e.g., 'Tang', 'Rococo').
        This implicitly represents the 'System Analyzes Database' step,
        as the system knows how to categorize elements.
        """
        if not style:
            return []
        return [elem for elem in self._elements if elem.get('style') == style]

    def get_database_summary(self) -> Dict[str, int]:
        """Provides a summary of element counts per style."""
        summary = {}
        for elem in self._elements:
            style = elem.get('style', 'Unknown')
            summary[style] = summary.get(style, 0) + 1
        return summary

# --- Part 2: Fusion Engine ---
# Contains the core logic for blending styles based on user input.

class FusionEngine:
    """Handles the logic for fusing design elements based on style ratios."""

    def __init__(self, database: ElementDatabase):
        """
        Initializes the Fusion Engine with a reference to the element database.
        """
        if not isinstance(database, ElementDatabase):
            raise ValueError("FusionEngine requires a valid ElementDatabase instance.")
        self._database = database
        print("融合引擎初始化完成，已连接到数据库。")

    def _validate_and_normalize_ratio(self, tang_perc: int, rococo_perc: int) -> Tuple[int, int]:
        """Validates and normalizes the user-provided percentages."""
        if not (0 <= tang_perc <= 100 and 0 <= rococo_perc <= 100):
            print("警告：输入百分比超出范围 [0, 100]，将尝试修正。")
            tang_perc = max(0, min(100, tang_perc))
            rococo_perc = max(0, min(100, rococo_perc))

        total = tang_perc + rococo_perc
        if total == 0:
            print("警告：百分比总和为 0，将重置为 50/50。")
            return 50, 50
        elif total != 100:
            print(f"警告：百分比总和 ({total}) 不为 100，将进行归一化。")
            norm_tang = round((tang_perc / total) * 100)
            norm_rococo = 100 - norm_tang
            print(f"  归一化后比例 - 唐: {norm_tang}%, 洛可可: {norm_rococo}%")
            return norm_tang, norm_rococo
        return tang_perc, rococo_perc

    def _select_weighted_elements(self,
                                 tang_elements: List[Dict],
                                 rococo_elements: List[Dict],
                                 tang_perc: int,
                                 rococo_perc: int,
                                 num_features: int) -> List[Dict]:
        """Selects elements based on the weighted ratio."""
        num_tang_target = round(num_features * (tang_perc / 100.0))
        num_rococo_target = num_features - num_tang_target

        num_tang_available = len(tang_elements)
        num_rococo_available = len(rococo_elements)

        num_tang_to_select = min(num_tang_target, num_tang_available)
        num_rococo_to_select = min(num_rococo_target, num_rococo_available)

        # Simple adjustment if one style is underrepresented but the other has surplus
        shortfall = num_features - (num_tang_to_select + num_rococo_to_select)
        if shortfall > 0:
            can_add_tang = num_tang_available - num_tang_to_select
            can_add_rococo = num_rococo_available - num_rococo_to_select
            # Prioritize adding from the style with higher percentage if possible
            if tang_perc >= rococo_perc:
                add_tang = min(shortfall, can_add_tang)
                add_rococo = min(shortfall - add_tang, can_add_rococo)
            else:
                add_rococo = min(shortfall, can_add_rococo)
                add_tang = min(shortfall - add_rococo, can_add_tang)
            num_tang_to_select += add_tang
            num_rococo_to_select += add_rococo

        print(f"根据权重计划选取: {num_tang_to_select} 个唐代元素, {num_rococo_to_select} 个洛可可元素。")

        selected_tang = random.sample(tang_elements, num_tang_to_select) if num_tang_to_select > 0 else []
        selected_rococo = random.sample(rococo_elements, num_rococo_to_select) if num_rococo_to_select > 0 else []

        # Combine and maybe shuffle for less predictable ordering in description
        selected_elements = selected_tang + selected_rococo
        random.shuffle(selected_elements) # Optional: shuffle the combined list
        return selected_elements

    def _generate_fusion_description(self,
                                     selected_elements: List[Dict],
                                     tang_perc: int,
                                     rococo_perc: int) -> str:
        """Generates a textual description of the fusion concept."""
        if not selected_elements:
            return "未能选定任何元素，无法生成描述。"

        tang_names = [e['name'] for e in selected_elements if e['style'] == 'Tang']
        rococo_names = [e['name'] for e in selected_elements if e['style'] == 'Rococo']

        description = f"一个融合设计概念，旨在体现 **{tang_perc}% 的唐代风韵** 与 **{rococo_perc}% 的洛可可情调**。\n\n"

        if tang_perc > rococo_perc + 10: # Tang dominant
            description += f"设计以唐代元素为主导，例如可能采用 **{'、'.join(tang_names)}** 的特征" if tang_names else "设计以唐代风格为主导"
            if rococo_names:
                description += f"，并精巧地融入了洛可可式的 **{'、'.join(rococo_names)}** 等元素作为点缀或对比。"
            else:
                 description += "，整体呈现东方古典美学。"
        elif rococo_perc > tang_perc + 10: # Rococo dominant
            description += f"设计侧重于洛可可风格，可能突出 **{'、'.join(rococo_names)}** 的特点" if rococo_names else "设计侧重于洛可可风格"
            if tang_names:
                description += f"，同时巧妙结合了 **{'、'.join(tang_names)}** 等唐代元素，增添异域或历史层次感。"
            else:
                description += "，整体展现西式宫廷的华丽与精致。"
        else: # Balanced
             description += "设计在唐代与洛可可风格间寻求和谐平衡。\n"
             features = []
             if tang_names: features.append(f"唐代的 **{'、'.join(tang_names)}**")
             if rococo_names: features.append(f"洛可可的 **{'、'.join(rococo_names)}**")
             if features:
                 description += f"尝试将 { ' 与 '.join(features) } 的特色有机结合"
             description += "，创造出一种新颖独特的跨文化美学体验。"

        return description

    def _generate_image_prompt(self, description: str, selected_elements: List[Dict], keywords: List[str]) -> str:
        """
        Generates a textual prompt suitable for an AI image generation model.
        THIS IS A PLACEHOLDER - does not actually generate an image.
        """
        element_names = [e['name'] for e in selected_elements]
        prompt = (
            f"Create a fashion design sketch illustrating: {description}. "
            f"Key visual elements to incorporate include: {', '.join(element_names)}. "
            f"Style keywords: {', '.join(keywords[:10])}. " # Limit keywords for clarity
            f"Emphasize the blend of Tang Dynasty and Rococo aesthetics according to the described ratio. "
            f"Focus on clothing details, silhouette, and mood."
            # Add more specifics if needed, e.g., color palette hints, fabric types etc.
        )
        return prompt

    def generate_fusion_concept(self,
                                tang_percentage: int,
                                rococo_percentage: int,
                                num_elements_to_feature: int = 5) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the fusion process: validates input, selects elements,
        generates description and image prompt.
        """
        print(f"\n--- 引擎开始处理: 唐={tang_percentage}%, 洛可可={rococo_percentage}%, 特征数={num_elements_to_feature} ---")

        # 1. Validate and Normalize Ratio
        tang_perc_norm, rococo_perc_norm = self._validate_and_normalize_ratio(tang_percentage, rococo_percentage)

        # 2. Get elements from DB (Represents System Analysis)
        tang_elements = self._database.get_elements_by_style('Tang')
        rococo_elements = self._database.get_elements_by_style('Rococo')

        if not tang_elements and tang_perc_norm > 0:
            print("警告：数据库缺少唐代元素，无法满足请求。")
        if not rococo_elements and rococo_perc_norm > 0:
            print("警告：数据库缺少洛可可元素，无法满足请求。")

        # 3. Select Weighted Elements (Core Fusion Logic Simulation)
        selected_elements = self._select_weighted_elements(
            tang_elements, rococo_elements, tang_perc_norm, rococo_perc_norm, num_elements_to_feature
        )

        if not selected_elements:
            print("错误：未能根据权重选出任何元素，融合失败。")
            return None

        # 4. Generate Fusion Description
        description = self._generate_fusion_description(selected_elements, tang_perc_norm, rococo_perc_norm)

        # 5. Extract Keywords
        combined_tags = set()
        for element in selected_elements:
            combined_tags.update(element.get("tags", []))
        keywords = list(combined_tags)

        # 6. Generate Image Prompt (Placeholder for AI Image Gen Call)
        image_prompt = self._generate_image_prompt(description, selected_elements, keywords)

        # 7. Package Output
        output = {
            "requested_ratio": f"唐:{tang_percentage}% / 洛可可:{rococo_percentage}%",
            "normalized_ratio": f"唐:{tang_perc_norm}% / 洛可可:{rococo_perc_norm}%",
            "fusion_description": description,
            "featured_elements": [f"{elem['name']} ({elem['style']})" for elem in selected_elements],
            "potential_keywords": keywords,
            "image_generation_prompt": image_prompt, # The prompt for the next step
            "generated_image_placeholder": f"[此处将是由AI根据以上prompt生成的图片]",
        }
        print("--- 融合概念生成成功 ---")
        return output


# --- Part 3: User Interface (Simulation) ---
# Simulates how a user might interact with the system.

class SimulatedUserInterface:
    """Simulates user interaction for selecting ratio and displaying results."""

    def get_user_style_ratio(self) -> Tuple[int, int]:
        """Simulates getting the style ratio from the user."""
        while True:
            try:
                print("\n--- 用户输入 ---")
                tang_str = input("请输入期望的唐代风格百分比 (0-100): ")
                tang_perc = int(tang_str)
                rococo_str = input("请输入期望的洛可可风格百分比 (0-100): ")
                rococo_perc = int(rococo_str)
                # Basic validation, more robust validation happens in the engine
                if 0 <= tang_perc <= 100 and 0 <= rococo_perc <= 100:
                    return tang_perc, rococo_perc
                else:
                    print("输入错误：百分比必须在 0 到 100 之间，请重试。")
            except ValueError:
                print("输入无效：请输入数字。")

    def display_fusion_result(self, result: Optional[Dict[str, Any]]):
        """Displays the fusion concept result to the user."""
        print("\n========== 融合设计概念输出 ==========")
        if result:
            print(f"请求比例: {result['requested_ratio']}")
            print(f"实际应用比例: {result['normalized_ratio']}")
            print("\n主要融合元素:")
            for element in result['featured_elements']:
                print(f"  - {element}")
            print("\n融合设计描述:")
            print(result['fusion_description'])
            # print("\n融合关键词:")
            # print(", ".join(result['potential_keywords'])) # Optional display
            print("\n--- AI图像生成提示 (模拟) ---")
            print(result['image_generation_prompt'])
            print("\n--- 模拟设计图 ---")
            print(result['generated_image_placeholder'])
        else:
            print("未能成功生成融合设计概念。")
        print("======================================")


# --- Main Execution ---
# Orchestrates the process using the defined classes.

if __name__ == "__main__":
    # 1. Initialize Database with sample data
    sample_data = [
        # (Add the element_database list from the previous example here)
        # 唐代仕女服饰元素
        {"id": "T001", "name": "交领襦裙", "style": "Tang", "type": "轮廓", "tags": ["优雅", "高腰", "经典廓形", "丝绸"]},
        {"id": "T002", "name": "披帛", "style": "Tang", "type": "配饰", "tags": ["轻盈", "装饰", "飘逸", "长丝巾"]},
        {"id": "T003", "name": "宝相花纹样", "style": "Tang", "type": "纹样", "tags": ["华丽", "对称", "佛教影响", "植物变形"]},
        {"id": "T004", "name": "凤纹刺绣", "style": "Tang", "type": "工艺", "tags": ["吉祥", "精致", "手工", "皇家象征"]},
        {"id": "T005", "name": "齐胸襦裙", "style": "Tang", "type": "轮廓", "tags": ["开放", "唐风", "高腰线", "丰满"]},
        {"id": "T006", "name": "圆领袍", "style": "Tang", "type": "轮廓", "tags": ["中性", "简洁", "常服", "胡服影响"]},
        {"id": "T007", "name": "团花纹样", "style": "Tang", "type": "纹样", "tags": ["圆形", "饱满", "装饰性", "织锦"]},
        {"id": "T008", "name": "帷帽", "style": "Tang", "type": "配饰", "tags": ["遮蔽", "神秘", "出行", "防沙"]},

        # 洛可可服饰元素
        {"id": "R001", "name": "紧身胸衣 (Corset)", "style": "Rococo", "type": "结构", "tags": ["塑形", "细腰", "束缚", "内搭"]},
        {"id": "R002", "name": "裙撑 (Pannier)", "style": "Rococo", "type": "结构", "tags": ["夸张", "体积感", "横向扩张", "支撑"]},
        {"id": "R003", "name": "蕾丝花边", "style": "Rococo", "type": "装饰", "tags": ["精致", "女性化", "边缘装饰", "繁复"]},
        {"id": "R004", "name": "蝴蝶结", "style": "Rococo", "type": "装饰", "tags": ["甜美", "装饰", "绸带", "点缀"]},
        {"id": "R005", "name": "卷草纹样 (Rocaille)", "style": "Rococo", "type": "纹样", "tags": ["曲线", "自然", "不对称", "装饰性强"]},
        {"id": "R006", "name": "粉彩配色", "style": "Rococo", "type": "色彩", "tags": ["柔和", "淡雅", "粉色系", "马卡龙色"]},
        {"id": "R007", "name": "萨克袍 (Watteau Gown)", "style": "Rococo", "type": "轮廓", "tags": ["宽松背部", "华托褶", "优雅", "宫廷"]},
        {"id": "R008", "name": "羽毛装饰", "style": "Rococo", "type": "装饰", "tags": ["轻盈", "奢华", "头饰", "扇子"]},
    ]
    db = ElementDatabase(initial_data=sample_data)
    print("数据库内容概要:", db.get_database_summary()) # Show that DB analysis happened

    # 2. Initialize Fusion Engine
    engine = FusionEngine(database=db)

    # 3. Initialize Simulated User Interface
    ui = SimulatedUserInterface()

    # 4. Get User Input (Style Ratio)
    tang_p, rococo_p = ui.get_user_style_ratio()

    # 5. Generate Fusion Concept using the Engine
    # Let's also specify how many key features we want the concept to focus on
    num_features_to_show = 5
    fusion_result = engine.generate_fusion_concept(
        tang_percentage=tang_p,
        rococo_percentage=rococo_p,
        num_elements_to_feature=num_features_to_show
    )

    # 6. Display the Result via the UI
    ui.display_fusion_result(fusion_result)

    print("\n--- 程序执行完毕 ---")

