import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;

@Getter
@Setter
public class Population {
    private ArrayList<Individual> individuals;

    public Population(int populationSize, int genomeLength, Main.ProblemInstance problemInstance) {
        individuals = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            double[] genome = new double[genomeLength];
            for (int j = 0; j < genomeLength; j++) {
                switch (problemInstance) {
                    case SCHWEFEL -> genome[j] = Math.random() * 1000 - 500; // [-500,500]
                    case SCHAFFER -> genome[j] = Math.random() * 200 - 100; // [-100,100]
                }
            }
            individuals.add(new Individual(genome, populationSize, i, problemInstance));
        }
    }

    public Population(Population other, Main.ProblemInstance problemInstance) {
        this.individuals = new ArrayList<>();
        for (Individual ind : other.getIndividuals()) {
            double[] copiedGenome = ind.getGenome().clone();
            this.individuals.add(new Individual(copiedGenome, ind.getAncestry().length(), problemInstance));
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < individuals.size(); i++) {
            sb.append("Individual ").append(i + 1).append(": ")
                    .append(individuals.get(i).toString())
                    .append("\n");
        }
        return sb.toString();
    }
}