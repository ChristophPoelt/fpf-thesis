import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class EvolutionaryAlgorithm {
    private final int POPULATION_SIZE;
    private final int GENERATIONS;
    private final double MUTATION_RATE;
    private final double RECOMBINATION_RATE;
    private final double MIGRATION_RATE;
    private final int GENOME_LENGTH;
    private final Main.ProblemInstance problemInstance;

    private Population population;
    private Population initialPopulation;

    private ArrayList<Double> bestFitnessHistory = new ArrayList<>();
    private List<Individual> bestIndividualsPerGeneration = new ArrayList<>();
    private final int SLIDING_WINDOW_SIZE = 10;
    private final double EPSILON = 0.0017;
    private final double DELTA = 0.005;
    private final double CD = 5.0;

    public enum StoppingCriterion {FIXED_NUMBER_OF_GENERATIONS, IMPROVEMENT_BASED, FITNESS_BASED, DIVERSITY_BASED, FIXED_TARGET_VALUE}

    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.FIXED_NUMBER_OF_GENERATIONS;
    private static StoppingCriterion stoppingCriterion = StoppingCriterion.IMPROVEMENT_BASED;
    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.FITNESS_BASED;
    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.DIVERSITY_BASED;
    //private static StoppingCriterion stoppingCriterion = StoppingCriterion.FIXED_TARGET_VALUE;

    public EvolutionaryAlgorithm(int populationSize, int generations, double mutationRate, double recombinationRate, double migrationRate, int genomeLength, Main.ProblemInstance problemInstance) {
        this.POPULATION_SIZE = populationSize;
        this.GENERATIONS = generations;
        this.MUTATION_RATE = mutationRate;
        this.RECOMBINATION_RATE = recombinationRate;
        this.MIGRATION_RATE = migrationRate;
        this.GENOME_LENGTH = genomeLength;
        this.problemInstance = problemInstance;
    }

    public RunResult runEvolutionaryAlgorithmLoop() {
        Population firstGeneration = new Population(POPULATION_SIZE, GENOME_LENGTH, problemInstance);
        firstGeneration.getIndividuals().sort((i1, i2) -> Double.compare(i1.getTargetValue(), i2.getTargetValue()));
        this.initialPopulation = new Population(firstGeneration, problemInstance);
        this.population = firstGeneration;
        int generation = 0;
        logGeneration(generation);

        while (generation < GENERATIONS) {
            population.getIndividuals().addAll(applyEvolutionaryOperators());
            cutOffSelection();

            if (shouldStop(generation)) {
                System.out.println(stoppingCriterion + " met at generation " + generation);
                break;
            }

            logGeneration(generation);
            trackBestIndividual();
            double bestFitness = population.getIndividuals().stream()
                    .mapToDouble(Individual::getTargetValue)
                    .min()
                    .orElse(Double.MAX_VALUE);
            bestFitnessHistory.add(bestFitness);

            generation++;
        }

        List<IndividualWithFPF> individualsWithFPF = calculateFPF();
        printBestIndividualsHistory();
        double finalBestFitness = population.getIndividuals().stream()
                .mapToDouble(Individual::getTargetValue)
                .min()
                .orElse(Double.MAX_VALUE);

        return new RunResult(individualsWithFPF, generation, finalBestFitness, bestIndividualsPerGeneration);
    }

    private boolean shouldStop(int generation) {
        return switch (stoppingCriterion) {
            case IMPROVEMENT_BASED -> checkImprovementBasedStoppingCriterion();
            case FITNESS_BASED -> checkFitnessBasedStoppingCriterion();
            case DIVERSITY_BASED -> checkDiversityBasedStoppingCriterion();
            case FIXED_TARGET_VALUE -> checkFixedTargetValue();
            default -> false;
        };
    }

    private void logGeneration(int generation) {
        System.out.printf("\n=== Generation %d ===\n", generation);
        for (Individual ind : population.getIndividuals()) {
            System.out.println(ind);
        }
        double bestFitness = population.getIndividuals().stream()
                .mapToDouble(Individual::getTargetValue)
                .min()
                .orElse(Double.MAX_VALUE);
        System.out.printf("Generation %d: Best Fitness = %.6f%n", generation, bestFitness);
    }

    public ArrayList<Individual> applyEvolutionaryOperators() {
        int mutationCount = (int) Math.ceil(MUTATION_RATE * POPULATION_SIZE);
        int recombinationCount = (int) Math.ceil(RECOMBINATION_RATE * POPULATION_SIZE);
        int migrationCount = (int) Math.ceil(MIGRATION_RATE * POPULATION_SIZE);

        Collections.shuffle(population.getIndividuals());
        ArrayList<Individual> offspring = new ArrayList<>();

        for (int i = 0; i < mutationCount; i++) {
            offspring.add(population.getIndividuals().get(i).mutate(problemInstance));
        }

        Collections.shuffle(population.getIndividuals());
        for (int i = 0; i < recombinationCount; i++) {
            offspring.addAll(population.getIndividuals().get(i).recombine(
                    population.getIndividuals().get(POPULATION_SIZE - i - 1), problemInstance));
        }

        for (int i = 0; i < migrationCount; i++) {
            offspring.add(migrate());
        }

        return offspring;
    }

    public Individual migrate() {
        double[] genome = new double[GENOME_LENGTH];
        for (int i = 0; i < GENOME_LENGTH; i++) {
            switch (problemInstance) {
                case SCHWEFEL -> genome[i] = Math.random() * 1000 - 500;
                case SCHAFFER -> genome[i] = Math.random() * 200 - 100;
            }
        }
        return new Individual(genome, 50, problemInstance);
    }

    public void cutOffSelection() {
        population.getIndividuals().sort((i1, i2) -> Double.compare(i1.getTargetValue(), i2.getTargetValue()));
        while (population.getIndividuals().size() > POPULATION_SIZE) {
            population.getIndividuals().remove(population.getIndividuals().size() - 1);
        }
    }

    private boolean checkImprovementBasedStoppingCriterion() {
        if (bestFitnessHistory.size() < SLIDING_WINDOW_SIZE) return false;
        double totalImprovement = 0.0;
        for (int i = 1; i < bestFitnessHistory.size(); i++) {
            totalImprovement += Math.abs(bestFitnessHistory.get(i) - bestFitnessHistory.get(i - 1));
        }
        return (totalImprovement / (bestFitnessHistory.size() - 1)) < EPSILON;
    }

    private boolean checkFitnessBasedStoppingCriterion() {
        double fBest = population.getIndividuals().stream().mapToDouble(Individual::getTargetValue).min().orElse(Double.MAX_VALUE);
        double fWorst = population.getIndividuals().stream().mapToDouble(Individual::getTargetValue).max().orElse(Double.MIN_VALUE);
        return (fWorst - fBest) < DELTA;
    }

    private boolean checkDiversityBasedStoppingCriterion() {
        double diversity = calculateDiversity();
        System.out.println("Current Diversity: " + diversity);
        return diversity < CD;
    }

    private double calculateDiversity() {
        List<Individual> inds = population.getIndividuals();
        double totalDistance = 0.0;
        int pairs = 0;

        for (int i = 0; i < inds.size(); i++) {
            for (int j = i + 1; j < inds.size(); j++) {
                double[] genome1 = inds.get(i).getGenome();
                double[] genome2 = inds.get(j).getGenome();
                double sum = 0.0;
                for (int k = 0; k < genome1.length; k++) {
                    sum += Math.pow(genome1[k] - genome2[k], 2);
                }
                totalDistance += Math.sqrt(sum);
                pairs++;
            }
        }
        return totalDistance / pairs;
    }

    private boolean checkFixedTargetValue() {
        switch (problemInstance) {
            case SCHWEFEL -> {
                return population.getIndividuals().stream().anyMatch(ind -> ind.getTargetValue() < 0.04);
            }
            case SCHAFFER -> {
                return population.getIndividuals().stream().anyMatch(ind -> ind.getTargetValue() < 0.13);
            }
            default -> {
                return false;
            }
        }
    }

    private void trackBestIndividual() {
        Individual best = population.getIndividuals().stream().min((i1, i2) -> Double.compare(i1.getTargetValue(), i2.getTargetValue())).orElse(null);
        if (best != null) bestIndividualsPerGeneration.add(best);
    }

    private void printBestIndividualsHistory() {
        System.out.println("\n=== Best Individuals of Each Generation ===");
        for (int i = 0; i < bestIndividualsPerGeneration.size(); i++) {
            System.out.printf("Generation %3d: %s%n", i, bestIndividualsPerGeneration.get(i));
        }
    }

    public List<IndividualWithFPF> calculateFPF() {
        List<IndividualWithFPF> individualsWithFPF = new ArrayList<>();
        int initialSize = initialPopulation.getIndividuals().size();

        for (int i = 0; i < initialSize; i++) {
            Individual initInd = initialPopulation.getIndividuals().get(i);
            int count = 0;
            double totalFitness = 0.0;
            for (Individual finalInd : population.getIndividuals()) {
                if (finalInd.getAncestry().get(i)) {
                    totalFitness += finalInd.getTargetValue();
                    count++;
                }
            }
            double fpf = (count > 0) ? totalFitness / count : 1.0;
            individualsWithFPF.add(new IndividualWithFPF(initInd, fpf));
        }

        return individualsWithFPF;
    }
}
