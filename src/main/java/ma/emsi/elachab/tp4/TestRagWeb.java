package ma.emsi.elachab.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRagWeb {

    // false : les deux sources sont consultées.
    // true  : Gemini choisit les sources pertinentes.
    private static final boolean ROUTAGE_INTELLIGENT = true;
    public static void main(String[] args) {

        // 1. Vérifier les clés API
        String geminiKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_API_KEY");

        if (geminiKey == null || geminiKey.isBlank()) {
            System.out.println("Erreur : GEMINI_KEY absente.");
            return;
        }

        if (tavilyKey == null || tavilyKey.isBlank()) {
            System.out.println("Erreur : TAVILY_API_KEY absente.");
            return;
        }

        // 2. Modèle Gemini
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.2)
                .logRequests(true)
                .logResponses(true)
                .build();

        // 3. Préparer les embeddings du PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30)
                        .split(document);

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(
                embeddingModel.embedAll(segments).content(),
                segments
        );

        System.out.println(
                "[INDEXATION] rag.pdf : "
                        + segments.size() + " segments"
        );

        // 4. Retriever documentaire
        ContentRetriever retrieverPdf =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 5. Moteur de recherche Tavily
        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(tavilyKey)
                        .build();

        // 6. Retriever Web
        ContentRetriever retrieverWeb =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .maxResults(3)
                        .build();

        // 7. Choix du mode de routage
        QueryRouter routeur;

        if (ROUTAGE_INTELLIGENT) {

            // Gemini sélectionne une ou plusieurs sources
            Map<ContentRetriever, String> descriptions =
                    new LinkedHashMap<>();

            descriptions.put(
                    retrieverPdf,
                    "Support de cours local sur le RAG, " +
                            "le fine-tuning, les embeddings, " +
                            "les bases vectorielles et LangChain4j. " +
                            "À utiliser pour les questions portant " +
                            "sur le contenu du support de cours."
            );

            descriptions.put(
                    retrieverWeb,
                    "Recherche sur Internet avec Tavily. " +
                            "À utiliser pour les actualités, " +
                            "les annonces récentes, les versions " +
                            "logicielles et les informations actuelles."
            );

            routeur = new LanguageModelQueryRouter(
                    model,
                    descriptions
            );

            System.out.println(
                    "[MODE] Routage intelligent activé"
            );

        } else {

            // Routage de base : PDF + Web
            routeur = new DefaultQueryRouter(
                    retrieverPdf,
                    retrieverWeb
            );

            System.out.println(
                    "[MODE] Recherche dans les deux sources"
            );
        }

        // 8. Journaliser les sources sélectionnées
        QueryRouter routeurAvecLogs = query -> {

            System.out.println(
                    "\n[ROUTAGE] Question : " + query.text()
            );

            Collection<ContentRetriever> selection =
                    routeur.route(query);

            if (selection.contains(retrieverPdf)) {
                System.out.println(
                        "[ROUTAGE] Source sélectionnée : rag.pdf"
                );
            }

            if (selection.contains(retrieverWeb)) {
                System.out.println(
                        "[ROUTAGE] Source sélectionnée : Web Tavily"
                );
            }

            if (selection.isEmpty()) {
                System.out.println(
                        "[ROUTAGE] Aucune source sélectionnée"
                );
            }

            return selection;
        };

        // 9. Retrieval Augmentor
        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(routeurAvecLogs)
                        .build();

        // 10. Assistant RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(
                        MessageWindowChatMemory.withMaxMessages(10)
                )
                .retrievalAugmentor(augmentor)
                .build();

        System.out.println(
                "\nTest 5 - RAG Web prêt"
        );

        // 11. Conversation interactive
        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.print(
                        "\nVotre question (fin pour quitter) : "
                );

                String question = scanner.nextLine();

                if ("fin".equalsIgnoreCase(question.trim())) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                try {
                    String reponse = assistant.chat(question);

                    System.out.println(
                            "\nAssistant : " + reponse
                    );
                } catch (Exception e) {
                    System.err.println(
                            "[ERREUR] " + e.getMessage()
                    );
                }
            }
        }
    }
}
