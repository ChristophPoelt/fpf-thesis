import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final int POPULATION_SIZE = 50;
    private static final int GENERATIONS = 50; // upper limit for the number of generations
    private static final double MUTATION_RATE = 0.1;
    private static final double RECOMBINATION_RATE = 0.15;
    private static final double MIGRATION_RATE = 0.1;
    private static final int RUNS = 500;
    private static final String JSON_FILE = "_hpi_schwefelFixedGenerations.json";

    private static List<Double> finalFitnessPhase1 = new ArrayList<>();
    private static List<Double> finalFitnessPhase2 = new ArrayList<>();

    public enum ProblemInstance {
        SCHWEFEL, HIMMELBLAU, H1, SCHAFFER
    }

    public static void main(String[] args) {
        ProblemInstance problem = ProblemInstance.SCHWEFEL;
        //ProblemInstance problem = ProblemInstance.HIMMELBLAU;
        //ProblemInstance problem = ProblemInstance.H1;
        //ProblemInstance problem = ProblemInstance.SCHAFFER;

        List<Individual> highPotentialIndividuals = new ArrayList<>();

        // PHASE 1: Nur HPI generieren
        for (int i = 0; i < RUNS; i++) {
            EvolutionaryAlgorithm ea = new EvolutionaryAlgorithm(POPULATION_SIZE, GENERATIONS, MUTATION_RATE, RECOMBINATION_RATE, MIGRATION_RATE, problem);
            RunResult runResult = ea.runEvolutionaryAlgorithmLoop();
            Individual hpi = ea.getHighPotentialIndividual();
            highPotentialIndividuals.add(hpi);
            finalFitnessPhase1.add(runResult.getFinalBestFitness());
        }

        // PHASE 2: Runs mit High Potential Individuals starten
        List<RunResult> results = new ArrayList<>();
        for (int i = 0; i < RUNS; i++) {
            Individual hpi = highPotentialIndividuals.get(i);
            EvolutionaryAlgorithm ea = new EvolutionaryAlgorithm(
                    POPULATION_SIZE, GENERATIONS, MUTATION_RATE, RECOMBINATION_RATE, MIGRATION_RATE, problem, hpi);
            RunResult result = ea.runEvolutionaryAlgorithmLoop();
            results.add(result);
            finalFitnessPhase2.add(result.getFinalBestFitness());
            System.out.println("Completed run " + (i + 1) + " with " + result.getTotalGenerations() + " generations.");
        }

        saveResultsToJson(results, JSON_FILE);

        double avgPhase1 = finalFitnessPhase1.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double avgPhase2 = finalFitnessPhase2.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double improvement = avgPhase1 - avgPhase2;

        System.out.println("\n=== Final Fitness Comparison ===");
        System.out.printf("Average Final Fitness – Phase 1 (Random Init): %.6f\n", avgPhase1);
        System.out.printf("Average Final Fitness – Phase 2 (with High Potential Individuals): %.6f\n", avgPhase2);
        System.out.printf("Improvement: %.6f\n", improvement);
        double percent = (improvement / avgPhase1) * 100.0;
        System.out.printf("Relative Improvement: %.2f %%\n", percent);

        double avgGenerations = results.stream().mapToInt(RunResult::getTotalGenerations).average().orElse(0);
        double avgFinalFitness = results.stream().mapToDouble(RunResult::getFinalBestFitness).average().orElse(0);
        System.out.printf("\nAverage Generations: %.2f\n", avgGenerations);
        System.out.printf("Average Final Best Fitness: %.6f\n", avgFinalFitness);
    }

    private static void saveResultsToJson(List<RunResult> results, String fileName) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(fileName), results);
            System.out.println("Run results saved to " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}