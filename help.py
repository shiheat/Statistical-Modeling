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
        print(f"���ݿ��ʼ����ɣ����� {len(self._elements)} ��Ԫ�ء�")

    def load_data(self, data: List[Dict[str, Any]]):
        """Loads element data into the database."""
        # In a real system, this might connect to a SQL/NoSQL DB and load.
        self._elements = data
        print(f"�ɹ����� {len(data)} ��Ԫ�ص����ݿ⡣")

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
        print("�ں������ʼ����ɣ������ӵ����ݿ⡣")

    def _validate_and_normalize_ratio(self, tang_perc: int, rococo_perc: int) -> Tuple[int, int]:
        """Validates and normalizes the user-provided percentages."""
        if not (0 <= tang_perc <= 100 and 0 <= rococo_perc <= 100):
            print("���棺����ٷֱȳ�����Χ [0, 100]��������������")
            tang_perc = max(0, min(100, tang_perc))
            rococo_perc = max(0, min(100, rococo_perc))

        total = tang_perc + rococo_perc
        if total == 0:
            print("���棺�ٷֱ��ܺ�Ϊ 0��������Ϊ 50/50��")
            return 50, 50
        elif total != 100:
            print(f"���棺�ٷֱ��ܺ� ({total}) ��Ϊ 100�������й�һ����")
            norm_tang = round((tang_perc / total) * 100)
            norm_rococo = 100 - norm_tang
            print(f"  ��һ������� - ��: {norm_tang}%, ��ɿ�: {norm_rococo}%")
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

        print(f"����Ȩ�ؼƻ�ѡȡ: {num_tang_to_select} ���ƴ�Ԫ��, {num_rococo_to_select} ����ɿ�Ԫ�ء�")

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
            return "δ��ѡ���κ�Ԫ�أ��޷�����������"

        tang_names = [e['name'] for e in selected_elements if e['style'] == 'Tang']
        rococo_names = [e['name'] for e in selected_elements if e['style'] == 'Rococo']

        description = f"һ���ں���Ƹ��ּ������ **{tang_perc}% ���ƴ�����** �� **{rococo_perc}% ����ɿ����**��\n\n"

        if tang_perc > rococo_perc + 10: # Tang dominant
            description += f"������ƴ�Ԫ��Ϊ������������ܲ��� **{'��'.join(tang_names)}** ������" if tang_names else "������ƴ����Ϊ����"
            if rococo_names:
                description += f"�������ɵ���������ɿ�ʽ�� **{'��'.join(rococo_names)}** ��Ԫ����Ϊ��׺��Աȡ�"
            else:
                 description += "��������ֶ����ŵ���ѧ��"
        elif rococo_perc > tang_perc + 10: # Rococo dominant
            description += f"��Ʋ�������ɿɷ�񣬿���ͻ�� **{'��'.join(rococo_names)}** ���ص�" if rococo_names else "��Ʋ�������ɿɷ��"
            if tang_names:
                description += f"��ͬʱ�������� **{'��'.join(tang_names)}** ���ƴ�Ԫ�أ������������ʷ��θС�"
            else:
                description += "������չ����ʽ��͢�Ļ����뾫�¡�"
        else: # Balanced
             description += "������ƴ�����ɿɷ���Ѱ���гƽ�⡣\n"
             features = []
             if tang_names: features.append(f"�ƴ��� **{'��'.join(tang_names)}**")
             if rococo_names: features.append(f"��ɿɵ� **{'��'.join(rococo_names)}**")
             if features:
                 description += f"���Խ� { ' �� '.join(features) } ����ɫ�л����"
             description += "�������һ����ӱ���صĿ��Ļ���ѧ���顣"

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
        print(f"\n--- ���濪ʼ����: ��={tang_percentage}%, ��ɿ�={rococo_percentage}%, ������={num_elements_to_feature} ---")

        # 1. Validate and Normalize Ratio
        tang_perc_norm, rococo_perc_norm = self._validate_and_normalize_ratio(tang_percentage, rococo_percentage)

        # 2. Get elements from DB (Represents System Analysis)
        tang_elements = self._database.get_elements_by_style('Tang')
        rococo_elements = self._database.get_elements_by_style('Rococo')

        if not tang_elements and tang_perc_norm > 0:
            print("���棺���ݿ�ȱ���ƴ�Ԫ�أ��޷���������")
        if not rococo_elements and rococo_perc_norm > 0:
            print("���棺���ݿ�ȱ����ɿ�Ԫ�أ��޷���������")

        # 3. Select Weighted Elements (Core Fusion Logic Simulation)
        selected_elements = self._select_weighted_elements(
            tang_elements, rococo_elements, tang_perc_norm, rococo_perc_norm, num_elements_to_feature
        )

        if not selected_elements:
            print("����δ�ܸ���Ȩ��ѡ���κ�Ԫ�أ��ں�ʧ�ܡ�")
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
            "requested_ratio": f"��:{tang_percentage}% / ��ɿ�:{rococo_percentage}%",
            "normalized_ratio": f"��:{tang_perc_norm}% / ��ɿ�:{rococo_perc_norm}%",
            "fusion_description": description,
            "featured_elements": [f"{elem['name']} ({elem['style']})" for elem in selected_elements],
            "potential_keywords": keywords,
            "image_generation_prompt": image_prompt, # The prompt for the next step
            "generated_image_placeholder": f"[�˴�������AI��������prompt���ɵ�ͼƬ]",
        }
        print("--- �ںϸ������ɳɹ� ---")
        return output


# --- Part 3: User Interface (Simulation) ---
# Simulates how a user might interact with the system.

class SimulatedUserInterface:
    """Simulates user interaction for selecting ratio and displaying results."""

    def get_user_style_ratio(self) -> Tuple[int, int]:
        """Simulates getting the style ratio from the user."""
        while True:
            try:
                print("\n--- �û����� ---")
                tang_str = input("�������������ƴ����ٷֱ� (0-100): ")
                tang_perc = int(tang_str)
                rococo_str = input("��������������ɿɷ��ٷֱ� (0-100): ")
                rococo_perc = int(rococo_str)
                # Basic validation, more robust validation happens in the engine
                if 0 <= tang_perc <= 100 and 0 <= rococo_perc <= 100:
                    return tang_perc, rococo_perc
                else:
                    print("������󣺰ٷֱȱ����� 0 �� 100 ֮�䣬�����ԡ�")
            except ValueError:
                print("������Ч�����������֡�")

    def display_fusion_result(self, result: Optional[Dict[str, Any]]):
        """Displays the fusion concept result to the user."""
        print("\n========== �ں���Ƹ������ ==========")
        if result:
            print(f"�������: {result['requested_ratio']}")
            print(f"ʵ��Ӧ�ñ���: {result['normalized_ratio']}")
            print("\n��Ҫ�ں�Ԫ��:")
            for element in result['featured_elements']:
                print(f"  - {element}")
            print("\n�ں��������:")
            print(result['fusion_description'])
            # print("\n�ںϹؼ���:")
            # print(", ".join(result['potential_keywords'])) # Optional display
            print("\n--- AIͼ��������ʾ (ģ��) ---")
            print(result['image_generation_prompt'])
            print("\n--- ģ�����ͼ ---")
            print(result['generated_image_placeholder'])
        else:
            print("δ�ܳɹ������ں���Ƹ��")
        print("======================================")


# --- Main Execution ---
# Orchestrates the process using the defined classes.

if __name__ == "__main__":
    # 1. Initialize Database with sample data
    sample_data = [
        # (Add the element_database list from the previous example here)
        # �ƴ���Ů����Ԫ��
        {"id": "T001", "name": "������ȹ", "style": "Tang", "type": "����", "tags": ["����", "����", "��������", "˿��"]},
        {"id": "T002", "name": "����", "style": "Tang", "type": "����", "tags": ["��ӯ", "װ��", "Ʈ��", "��˿��"]},
        {"id": "T003", "name": "���໨����", "style": "Tang", "type": "����", "tags": ["����", "�Գ�", "���Ӱ��", "ֲ�����"]},
        {"id": "T004", "name": "���ƴ���", "style": "Tang", "type": "����", "tags": ["����", "����", "�ֹ�", "�ʼ�����"]},
        {"id": "T005", "name": "������ȹ", "style": "Tang", "type": "����", "tags": ["����", "�Ʒ�", "������", "����"]},
        {"id": "T006", "name": "Բ����", "style": "Tang", "type": "����", "tags": ["����", "���", "����", "����Ӱ��"]},
        {"id": "T007", "name": "�Ż�����", "style": "Tang", "type": "����", "tags": ["Բ��", "����", "װ����", "֯��"]},
        {"id": "T008", "name": "�ñ", "style": "Tang", "type": "����", "tags": ["�ڱ�", "����", "����", "��ɳ"]},

        # ��ɿɷ���Ԫ��
        {"id": "R001", "name": "�������� (Corset)", "style": "Rococo", "type": "�ṹ", "tags": ["����", "ϸ��", "����", "�ڴ�"]},
        {"id": "R002", "name": "ȹ�� (Pannier)", "style": "Rococo", "type": "�ṹ", "tags": ["����", "�����", "��������", "֧��"]},
        {"id": "R003", "name": "��˿����", "style": "Rococo", "type": "װ��", "tags": ["����", "Ů�Ի�", "��Եװ��", "����"]},
        {"id": "R004", "name": "������", "style": "Rococo", "type": "װ��", "tags": ["����", "װ��", "���", "��׺"]},
        {"id": "R005", "name": "������� (Rocaille)", "style": "Rococo", "type": "����", "tags": ["����", "��Ȼ", "���Գ�", "װ����ǿ"]},
        {"id": "R006", "name": "�۲���ɫ", "style": "Rococo", "type": "ɫ��", "tags": ["���", "����", "��ɫϵ", "����ɫ"]},
        {"id": "R007", "name": "������ (Watteau Gown)", "style": "Rococo", "type": "����", "tags": ["���ɱ���", "������", "����", "��͢"]},
        {"id": "R008", "name": "��ëװ��", "style": "Rococo", "type": "װ��", "tags": ["��ӯ", "�ݻ�", "ͷ��", "����"]},
    ]
    db = ElementDatabase(initial_data=sample_data)
    print("���ݿ����ݸ�Ҫ:", db.get_database_summary()) # Show that DB analysis happened

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

    print("\n--- ����ִ����� ---")

