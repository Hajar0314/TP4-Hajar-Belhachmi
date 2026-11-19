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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * Test 5 : RAG avec récupération sur le Web (Tavily) en plus du PDF.
 * Bonus : le PDF n'est pas utilisé si la question ne porte pas sur son contenu.
 */
public class Test4 {

    public static void main(String[] args) {
        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null) {
            System.out.println("La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }
        String tavilyKey = System.getenv("TAVILY_KEY");
        if (tavilyKey == null) {
            System.out.println("La variable d'environnement TAVILY_KEY n'est pas définie.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // Phase 1 : le PDF (comme le RAG naïf)
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore = ingerer("rag.pdf", embeddingModel);

        ContentRetriever contentRetrieverPdf = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Moteur de recherche Web Tavily + son ContentRetriever
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever contentRetrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        // ===== CODE INITIAL (mis en commentaire pour le correcteur) =====
        // Problème : le DefaultQueryRouter envoie TOUJOURS la question aux 2 retrievers,
        // donc des extraits du PDF sont ajoutés même si la question n'a aucun rapport.
        // QueryRouter queryRouter = new DefaultQueryRouter(contentRetrieverPdf, contentRetrieverWeb);

        // ===== CODE CORRIGÉ =====
        // Modèle dédié au routage : température 0 pour une réponse stable
        ChatModel modelRouteur = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash")
                .temperature(0.0)
                .logRequestsAndResponses(true)
                .build();

        // Classe interne locale : après la définition de modelRouteur, contentRetrieverPdf et contentRetrieverWeb
        class QueryRouterPdfOuWeb implements QueryRouter {

            private final PromptTemplate promptTemplate = PromptTemplate.from(
                    "Est-ce que la requête '{{requete}}' porte sur l'IA, et plus précisément sur le 'RAG' "
                            + "(Retrieval Augmented Generation) ou le 'Fine Tuning' ? "
                            + "Réponds 'oui' seulement si la requête porte sur l'IA, le RAG ou le Fine Tuning. "
                            + "Réponds 'non' si elle porte sur un autre sujet (sport, actualité, météo, culture générale...). "
                            + "Réponds 'peut-être' si tu hésites. "
                            + "Réponds uniquement par 'oui', 'non' ou 'peut-être'.");

            @Override
            public List<ContentRetriever> route(Query query) {
                String question = promptTemplate.apply(Map.of("requete", query.text())).text();
                String reponse = modelRouteur.chat(question).trim().toLowerCase();
                if (reponse.startsWith("non")) {
                    // Sans rapport avec le PDF : on ne consulte que le Web
                    return Collections.singletonList(contentRetrieverWeb);
                } else {
                    // Lié au cours (ou doute) : PDF + Web
                    return List.of(contentRetrieverPdf, contentRetrieverWeb);
                }
            }
        }

        QueryRouter queryRouter = new QueryRouterPdfOuWeb();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        conversationAvec(assistant);
    }

    /** Phase 1 : charge un fichier du classpath, le découpe, calcule les embeddings et les enregistre. */
    private static EmbeddingStore<TextSegment> ingerer(String nomFichier, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, documentParser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
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