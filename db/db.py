from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_PATH = "sqlite:///db/finanziamenti.db"
engine = create_engine(DB_PATH, echo=False)
Base = declarative_base()

class RichiestaFinanziamento(Base):
    __tablename__ = "richieste_finanziamento"
    RichiestaFinanziamentoID = Column(Integer, primary_key=True, unique=True, nullable=False)
    Eta = Column(Integer)
    Sesso = Column(String)
    TitoloStudio = Column(String)
    RedditoLordoUltimoAnno = Column(Float)
    AnniEsperienzaLavorativa = Column(Integer)
    InformazioniImmobile = Column(String)
    ImportoRichiesto = Column(Float)
    ScopoFinanziamento = Column(String)
    TassoInteresseFinanziamento = Column(Float)
    ImportoRichiestoDivisoReddito = Column(Float)
    DurataDellaStoriaCreditiziaInAnni = Column(Integer)
    AffidabilitàCreditizia = Column(Integer)
    InadempienzeFinanziamentiPrecedenti = Column(String)
    ProbabilitaFinanziamentoApprovato = Column(Float)

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)