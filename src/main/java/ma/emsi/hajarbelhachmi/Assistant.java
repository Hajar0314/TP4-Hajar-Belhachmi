package ma.emsi.hajarbelhachmi;

/**
 * Assistant conversationnel utilisé par tous les tests du TP 4.
 * L'implémentation est générée par LangChain4j (AiServices).
 */
public interface Assistant {
    // Prend un message de l'utilisateur et retourne la réponse du LLM.
    String chat(String userMessage);
}