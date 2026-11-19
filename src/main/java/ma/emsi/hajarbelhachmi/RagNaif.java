package ma.emsi.hajarbelhachmi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

/**
 * Test 1 : RAG "naïf" avec les étapes explicites (pas de RAG facile).
 */
public class RagNaif {

    public static void main(String[] args) {
        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null) {
            System.out.println("La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Température basse : le RAG sert à contrôler l'exactitude des réponses.
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash")
                .temperature(0.3)
                // .logRequestsAndResponses(true)   // <-- à décommenter pour l'étape "avec logging"
                .build();

        // ===== Phase 1 : enregistrement des embeddings =====

        // 1. Création du Document à partir du PDF (ressource du classpath), avec le parseur Tika
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);

        // 2. Découpage en segments : 300 caractères max, 30 de chevauchement
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // 3. Modèle d'embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 4. Embeddings des segments (embedAll retourne une Response<List<Embedding>>)
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 5. Enregistrement dans le magasin d'embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 : utilisation des embeddings pour répondre aux questions =====

        // 6. ContentRetriever : les 2 résultats les plus pertinents, score >= 0,5
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 7. Mémoire de 10 messages
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // 8. Assistant, avec le ContentRetriever en plus
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        // 9. Conversation
        conversationAvec(assistant);
    }

    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}