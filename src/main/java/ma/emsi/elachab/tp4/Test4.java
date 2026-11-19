
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

public class Test4 {

    public static void main(String[] args) {

        // ==================================================
        // 1. Vérification des clés API
        // ==================================================

        String geminiKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_API_KEY");

        if (geminiKey == null || geminiKey.isBlank()) {
            System.out.println("Erreur : GEMINI_KEY non définie.");
            return;
        }

        if (tavilyKey == null || tavilyKey.isBlank()) {
            System.out.println("Erreur : TAVILY_API_KEY non définie.");
            return;
        }

        // ==================================================
        // 2. Création du ChatModel Gemini
        // ==================================================

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.2)
                .logRequests(true)
                .logResponses(true)
                .build();

        // ==================================================
        // 3. Chargement et découpage du document PDF
        // ==================================================

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30)
                        .split(document);

        // ==================================================
        // 4. Embeddings et magasin vectoriel
        // ==================================================

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(
                embeddingModel.embedAll(segments).content(),
                segments
        );

        System.out.println(
                "[INDEXATION] rag.pdf : " +
                        segments.size() + " segments"
        );

        // ==================================================
        // 5. Retriever documentaire (PDF)
        // ==================================================

        ContentRetriever retrieverPdf =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // ==================================================
        // 6. Moteur de recherche Tavily
        // ==================================================

        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(tavilyKey)
                        .build();

        // ==================================================
        // 7. Retriever Web
        // ==================================================

        ContentRetriever retrieverWeb =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .maxResults(3)
                        .build();


        // ==================================================
        // 8. ANCIENNE VERSION : DefaultQueryRouter
        // ==================================================
        // Cette version interroge systématiquement les
        // deux sources : PDF et Web.
        //
        // Problème : pour une question sur l'actualité,
        // des extraits inutiles du PDF sont également
        // ajoutés au prompt envoyé à Gemini.
        //
        // Ancien code conservé pour l'exercice bonus :

        /*
        QueryRouter routeur = new DefaultQueryRouter(
                retrieverPdf,
                retrieverWeb
        );
        */


        // ==================================================
        // 9. ROUTAGE : NOUVELLE VERSION INTELLIGENTE
        // ==================================================

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
                        "les événements récents, les versions " +
                        "logicielles et les informations actuelles."
        );

        QueryRouter routeur =
                new LanguageModelQueryRouter(
                        model,
                        descriptions
                );

        System.out.println(
                "[MODE] Routage intelligent activé"
        );

        // ==================================================
        // 10. Logs des sources sélectionnées
        // ==================================================

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

        // ==================================================
        // 11. Configuration du RetrievalAugmentor
        // ==================================================

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(routeurAvecLogs)
                        .build();

        // ==================================================
        // 12. Création de l'assistant
        // ==================================================

        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(
                                MessageWindowChatMemory
                                        .withMaxMessages(10)
                        )
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        System.out.println(
                "\nTest 5 - RAG Web avec Tavily prêt"
        );

        // ==================================================
        // 13. Conversation interactive
        // ==================================================

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
