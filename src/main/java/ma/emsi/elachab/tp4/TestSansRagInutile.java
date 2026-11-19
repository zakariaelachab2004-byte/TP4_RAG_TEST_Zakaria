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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;

public class TestSansRagInutile {

    public static void main(String[] args) {

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            System.out.println("Erreur : GEMINI_KEY non définie.");
            return;
        }

        // 1. Modèle Gemini
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.2)
                .logRequests(true)
                .logResponses(true)
                .build();

        // 2. Chargement du support de cours
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30)
                        .split(document);

        // 3. Embeddings et stockage vectoriel
        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(
                embeddingModel.embedAll(segments).content(),
                segments
        );

        System.out.println(
                "Nombre de segments : " + segments.size()
        );

        // 4. Retriever documentaire
        ContentRetriever retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 5. Prompt de décision
        PromptTemplate template = PromptTemplate.from("""
                Tu es un système de décision pour un assistant RAG.

                Détermine si la question doit utiliser le support
                de cours sur le RAG, le fine-tuning et LangChain4j.

                Réponds uniquement par l'un de ces mots :
                oui, non, peut-être.

                Règles :
                - oui : question explicitement liée au support de cours.
                - non : salutation, connaissance générale ou
                  question sans rapport avec le support.
                - peut-être : question ambiguë dont le besoin
                  documentaire est incertain.

                Question : {{question}}

                Décision :
                """);

        // 6. Routeur personnalisé
        QueryRouter routeur = query -> {

            String question = query.text();

            String prompt = template.apply(
                    Map.of("question", question)
            ).text();

            System.out.println(
                    "\n[DECISION] Question : " + question
            );

            String decision = model.chat(prompt)
                    .trim()
                    .toLowerCase(Locale.ROOT)
                    .replaceAll("[.!?]", "");

            System.out.println(
                    "[DECISION] Réponse Gemini : " + decision
            );

            if (decision.equals("oui")) {
                System.out.println(
                        "[ROUTAGE] RAG activé : consultation de rag.pdf"
                );

                return List.of(retriever);
            }

            if (decision.equals("non")) {
                System.out.println(
                        "[ROUTAGE] RAG désactivé : réponse directe"
                );

                return List.of();
            }

            // En cas de doute, on consulte le document.
            System.out.println(
                    "[ROUTAGE] Décision incertaine : RAG activé"
            );

            return List.of(retriever);
        };

        // 7. RAG avancé avec routeur personnalisé
        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(routeur)
                        .build();

        // 8. Assistant conversationnel
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(
                        MessageWindowChatMemory.withMaxMessages(10)
                )
                .retrievalAugmentor(augmentor)
                .build();

        System.out.println(
                "\nTest 4 - Désactivation intelligente du RAG"
        );

        // 9. Questions interactives
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

                String reponse = assistant.chat(question);

                System.out.println(
                        "\nAssistant : " + reponse
                );
            }
        }
    }
}
