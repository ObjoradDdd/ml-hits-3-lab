package hits.ml.lab3.service;

public interface TranslatorService {
    String translate(Byte[] text, Byte[] sourceLanguage);
}
