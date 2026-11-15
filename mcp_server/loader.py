import json
from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CustomLoader(BaseLoader):
    def __init__(self, path):
        self.path = path

    def lazy_load(self) -> Iterator[Document]:
        path = Path(self.path)
        with open(path, "r") as f:
            for l in f:
                if not l:
                    continue
                doc = json.loads(l)
                doc["metadata"]["orig_elements"] = doc["element_id"]
                yield Document(page_content=doc["text"], metadata=doc["metadata"])


def loader(
    paths: list[str] | None = None,
    graph_ingestion: bool = False,
):
    dataset = []
    if paths is not None:
        for path in paths:
            loader = CustomLoader(path)
            data = loader.load()
            if graph_ingestion:
                data = " ".join(
                    [
                        doc.page_content
                        for doc in data
                        if doc.metadata["orig_elements"].split("_")[-2] == "child"
                    ]
                )

            dataset.append(data)
            return dataset
    else:
        loader = DirectoryLoader(
            ".data/.Cleaned",
            loader_cls=CustomLoader,  # type: ignore
            show_progress=True,
        )
        data = loader.load()
        return data
