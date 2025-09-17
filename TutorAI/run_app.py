from tutorai.dashboard import TutorAIDashboard
from tutorai.vector_index import ChunkingConfig
from tutorai.config import DEFAULT_BOOK_URL

DEFAULT_BOOK_URL = "https://github.com/infoalpha/Data-Science-books/blob/master/storytelling-with-data-cole-nussbaumer-knaflic.pdf"
dashboard = TutorAIDashboard(
    default_book_url=DEFAULT_BOOK_URL,
    chunk_cfg=ChunkingConfig(chunk_size=900, chunk_overlap=150),)
app = dashboard.build_app()
app.launch()
