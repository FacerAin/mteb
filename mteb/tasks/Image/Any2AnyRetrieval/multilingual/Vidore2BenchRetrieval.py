from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "french": ["fra-Latn"],
    "spanish": ["spa-Latn"],
    "english": ["eng-Latn"],
    "german": ["deu-Latn"],
    "korean": ["kor-Hang"],
}


def _load_data(
    path: str,
    splits: str,
    langs: list | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    if langs is None:
        corpus = {}
        queries = {}
        relevant_docs = {}
    else:
        corpus = {lang: {} for lang in langs}
        queries = {lang: {} for lang in langs}
        relevant_docs = {lang: {} for lang in langs}

    for split in splits:
        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['query-id']}",
                "text": x["query"],
                "image": None,
                "modality": "text",
            },
            remove_columns=["query-id", "query"],
        )

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['corpus-id']}",
                "text": None,
                "modality": "image",
            },
            remove_columns=["corpus-id"],
        )

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )

        if langs is None:
            queries[split] = query_ds
            corpus[split] = corpus_ds
            relevant_docs[split] = {}
            for row in qrels_ds:
                qid = f"query-{split}-{row['query-id']}"
                did = f"corpus-{split}-{row['corpus-id']}"
                if qid not in relevant_docs[split]:
                    relevant_docs[split][qid] = {}
                relevant_docs[split][qid][did] = int(row["score"])
        else:
            for lang in langs:
                queries[lang][split] = query_ds.filter(lambda x: x["language"] == lang)

                corpus[lang][split] = corpus_ds

                relevant_docs[lang][split] = {}
                for row in qrels_ds:
                    qid = f"query-{split}-{row['query-id']}"
                    did = f"corpus-{split}-{row['corpus-id']}"
                    if qid not in relevant_docs[lang][split]:
                        relevant_docs[lang][split][qid] = {}
                    relevant_docs[lang][split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


class Vidore2ESGReportsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2ESGReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/esg_reports_v2",
            "revision": "0542c0d03da0ec1c8cbc517c8d78e7e95c75d3d3",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 30,
                    "num_queries": 228,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class Vidore2EconomicsReportsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2EconomicsReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/economics_reports_v2",
            "revision": "b3e3a04b07fbbaffe79be49dabf92f691fbca252",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 5,
                    "num_queries": 232,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class Vidore2BioMedicalLecturesRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2BioMedicalLecturesRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/biomedical_lectures_v2",
            "revision": "a29202f0da409034d651614d87cd8938d254e2ea",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 27,
                    "num_queries": 640,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class Vidore2ESGReportsHLRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2ESGReportsHLRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/esg_reports_human_labeled_v2",
            "revision": "6d467dedb09a75144ede1421747e47cf036857dd",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 27,
                    "num_queries": 640,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class KoVidoreMIRRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreMIRRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "whybe-choi/kovidore-mir-v1.0-beir",
            "revision": "8a69ee34933fa2fb32ee5a20b1f537e99a22529f",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2, # TODO: update this bibtex
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."}, # TODO: Is this prompt correct?
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1366,
                    "num_queries": 1496,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True



class KoVidoreVQARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "whybe-choi/kovidore-vqa-v1.0-beir",
            "revision": "65771fa2bdd649c77faaab1b67378d5a782ae547",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2, # TODO: update this bibtex
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."}, # TODO: Is this prompt correct?
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1101,
                    "num_queries": 1500,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True

class KoVidoreSlideRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreSlideRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "whybe-choi/kovidore-slide-v1.0-beir",
            "revision": "6d6c1727766d44434bb4a026724adbab67b41001",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2, # TODO: update this bibtex
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."}, # TODO: Is this prompt correct?
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1415,
                    "num_queries": 180,
                    "average_relevant_docs_per_query": 1.2444,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True

class KoVidoreOfficeRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreOfficeRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "whybe-choi/kovidore-office-v1.0-beir",
            "revision": "00faa811db01ab3dd1e5121fe60c85790aa93f42",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2, # TODO: update this bibtex
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."}, # TODO: Is this prompt correct?
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1993,
                    "num_queries": 222,
                    "average_relevant_docs_per_query": 1.0991,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
        
class KoVidoreFinOCRRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreFinOCRRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "whybe-choi/kovidore-finocr-v1.0-beir",
            "revision": "0959e3bed492ecac92391ae01cbab1cd73725d28",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2, # TODO: update this bibtex
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."}, # TODO: Is this prompt correct?
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 2000,
                    "num_queries": 187,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True