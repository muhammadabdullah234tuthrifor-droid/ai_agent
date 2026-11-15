import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
)
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_from_dicts

from logger import get_logger

logger = get_logger("data_prep")


@dataclass
class UltimateDataPrep:
    path: str
    tokenizer: AutoTokenizer
    p_chunk_size: int = 128
    p_chunk_overlap: int = 8
    c_chunk_size: int = 64
    c_chunk_overlap: int = 4
    min_threshold: float = 0.20
    max_threshold: float = 0.30
    chunk_len: int = 100
    min_occurrences: int = 5

    def yolo(self):

        self.cleaner(self.chunker(self.remover(self.parser())))

    @staticmethod
    def recursive_token_split(
        element: dict,
        tokenizer,
        parent_chunk_size: int = 128,
        parent_overlap: int = 8,
        child_chunk_size: int = 64,
        child_overlap: int = 0,
    ) -> tuple[list[dict], list[dict]]:

        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        text = element.get("text", "")
        base_id = element.get("element_id", "chunk")

        if not text:
            return [deepcopy(element)], []

        def token_count(t: str) -> int:
            return len(tokenizer.encode(t, add_special_tokens=False))

        def add_overlap(original_chunks: list[str], c_overlap: int) -> list[str]:
            if len(original_chunks) <= 1 or c_overlap <= 0:
                return original_chunks

            result = [original_chunks[0]]
            for i in range(1, len(original_chunks)):
                prev_tokens = tokenizer.encode(
                    original_chunks[i - 1], add_special_tokens=False
                )
                overlap_tokens = prev_tokens[-c_overlap:]
                overlap_text = tokenizer.decode(overlap_tokens)

                # FIX: Prevent missing spaces when concatenating decoded tokens
                if (
                    overlap_text
                    and not overlap_text.endswith(" ")
                    and not original_chunks[i].startswith(" ")
                ):
                    overlap_text += " "

                result.append(overlap_text + original_chunks[i])

            return result

        def split_text(
            t: str, c_size: int, c_overlap: int, current_separators: list[str]
        ) -> list[str]:
            if token_count(t) <= c_size:
                return [t]

            if not current_separators or current_separators[0] == "":
                # Hard split fallback
                tokens = tokenizer.encode(t, add_special_tokens=False)
                step = max(1, c_size - c_overlap)
                chunks = []
                for i in range(0, len(tokens), step):
                    chunk_tokens = tokens[i : i + c_size]
                    chunks.append(tokenizer.decode(chunk_tokens))
                return chunks

            separator = current_separators[0]
            pieces = t.split(separator)

            if len(pieces) == 1:
                return split_text(t, c_size, c_overlap, current_separators[1:])

            chunks = []
            current = ""

            for piece in pieces:
                candidate = piece if current == "" else current + separator + piece

                if token_count(candidate) <= c_size:
                    current = candidate
                else:
                    if current:
                        chunks.extend(
                            split_text(
                                current, c_size, c_overlap, current_separators[1:]
                            )
                        )
                    current = piece

            if current:
                chunks.extend(
                    split_text(current, c_size, c_overlap, current_separators[1:])
                )

            return add_overlap(chunks, c_overlap)

        parent_texts = split_text(text, parent_chunk_size, parent_overlap, separators)

        parent_records = []
        child_records = []

        for i, p_text in enumerate(parent_texts):
            parent_element = deepcopy(element)
            parent_id = f"{base_id}_parent_{i}"

            parent_element["element_id"] = parent_id
            parent_element["text"] = p_text
            parent_records.append(parent_element)

            child_texts = split_text(
                p_text, child_chunk_size, child_overlap, separators
            )
            for j, c_text in enumerate(child_texts):
                child_element = deepcopy(parent_element)
                child_element["element_id"] = f"{parent_id}_child_{j}"
                child_element["text"] = c_text
                child_records.append(child_element)

        return (parent_records, child_records)

    def parser(self):
        path = Path(self.path)
        resolved_path = path.resolve()
        filename = path.stem + ".jsonl"
        new_path = resolved_path.parent.parent / ".data/.Clean"
        new_file_path = new_path / filename
        elements = []
        try:
            # print("loading elements of doc...")
            with open(new_file_path, "r") as f:
                elements = elements_from_dicts([json.loads(d) for d in f])
            logger.debug(
                "elements of the given file: %s", [ele.to_dict() for ele in elements]
            )
        except FileNotFoundError:
            # print("creating elements of doc...")
            elements = partition(filename=str(path), strategy="auto")
            logger.debug(
                "elements of the given file: %s", [ele.to_dict() for ele in elements]
            )
            Path(new_path).mkdir(exist_ok=True)
            with open(new_file_path, "w") as f:
                for el in elements:
                    f.write(json.dumps(el.to_dict()) + "\n")

        return elements

    def remover(self, elements):
        # print("removing  noise(repetitive text)...")
        min_threshold = self.min_threshold
        max_threshold = self.max_threshold
        min_occurrences = self.min_occurrences
        chunk_len = self.chunk_len
        line_counts = Counter()
        total_elements = len(elements)
        logger.debug("length of elements: %i", total_elements)
        threshold = max(
            min_threshold, min(max_threshold, min_occurrences / total_elements)
        )

        for ele in elements:
            lines = [line.strip() for line in ele.text.split("\n") if line.strip()]
            unique_lines_in_chunk = set(lines)
            for line in unique_lines_in_chunk:
                line_counts[line] += 1

        noise_lines = {
            line
            for line, count in line_counts.items()
            if (count / total_elements) > threshold and len(line) < chunk_len
        }
        logger.debug("noisy lines: %s", noise_lines)

        for ele in elements:
            lines = ele.text.split("\n")
            cleaned_lines = [line for line in lines if line.strip() not in noise_lines]
            ele.text = "\n".join(cleaned_lines)
        logger.debug(
            "elements without repetitive noise: %s", [ele.to_dict() for ele in elements]
        )
        return elements

    def chunker(self, cleaned_elements):
        tokenizer = self.tokenizer
        p_chunk_size = self.p_chunk_size
        p_chunk_overlap = self.p_chunk_overlap
        c_chunk_size = self.c_chunk_size
        c_chunk_overlap = self.c_chunk_overlap
        # print("Intelligent chunking...")
        chunks = chunk_by_title(
            elements=cleaned_elements,
            max_characters=8000,
            new_after_n_chars=8000,
            combine_text_under_n_chars=200,
        )
        logger.debug(
            "chunks after Intelligent chunking: %s",
            [chunk.to_dict() for chunk in chunks],
        )
        logger.debug("total chunks: %i", len(chunks))
        # print("recusive chunking(token aware)...")
        cchunks = [
            self.recursive_token_split(
                c.to_dict(),
                tokenizer,
                p_chunk_size,
                p_chunk_overlap,
                c_chunk_size,
                c_chunk_overlap,
            )
            for c in chunks
        ]
        logger.debug("chunks after recursive chunking(token aware): %s", cchunks)
        cld_chk = [t for p in cchunks for t in p[1]]
        pt_chk = [t for p in cchunks for t in p[0]]

        logger.debug("child chunks: %s", cld_chk)
        logger.debug("parent chunks: %s", pt_chk)

        cld_els = elements_from_dicts(cld_chk)
        pt_els = elements_from_dicts(pt_chk)

        return (pt_els, cld_els)

    def cleaner(self, elements):
        cld_els = elements[1]
        pt_els = elements[0]
        # print("cleaning chunks...")
        path = Path(self.path)
        resolved_path = Path.resolve(path)
        filename = path.stem + "_.jsonl"
        new_path = resolved_path.parent.parent / ".data/.Cleaned"
        new_file_path = new_path / filename
        cleaned_cld_chks = []
        cleaned_pt_chks = []
        logger.debug("creating chunks of doc...")
        for el in cld_els:
            cleaned_cld_text = el.text
            cleaned_cld_text = clean_extra_whitespace(cleaned_cld_text)
            cleaned_cld_text = group_broken_paragraphs(cleaned_cld_text)
            cleaned_cld_text = clean_non_ascii_chars(cleaned_cld_text)
            el.text = cleaned_cld_text
            cleaned_cld_chks.append(el)
        for el in pt_els:
            cleaned_pt_text = el.text
            cleaned_pt_text = clean_extra_whitespace(cleaned_pt_text)
            cleaned_pt_text = group_broken_paragraphs(cleaned_pt_text)
            cleaned_pt_text = clean_non_ascii_chars(cleaned_pt_text)
            el.text = cleaned_pt_text
            cleaned_pt_chks.append(el)
        logger.debug(
            "cleaned child chunks: %s", [chk.to_dict() for chk in cleaned_cld_chks]
        )
        logger.debug(
            "cleaned parent chunks: %s", [chk.to_dict() for chk in cleaned_pt_chks]
        )

        logger.debug("Before cleaning parent eles: %s", len(pt_els))
        logger.debug("After cleaning parent eles:  %s", len(cleaned_pt_chks))
        logger.debug("Before cleaning child eles: %s", len(cld_els))
        logger.debug("After cleaning child eles:  %s", len(cleaned_cld_chks))
        Path(new_path).mkdir(exist_ok=True)
        with open(new_file_path, "w") as f:
            for chk in cleaned_cld_chks:
                f.write(json.dumps(chk.to_dict()) + "\n")
            for chk in cleaned_pt_chks:
                f.write(json.dumps(chk.to_dict()) + "\n")
