import lombok.Getter;
import lombok.Setter;
import java.util.ArrayList;

@Getter
@Setter
public class Population {
    private ArrayList<Individual> individuals;

    public Population(int populationSize, Main.ProblemInstance problemInstance) {
        individuals = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            switch (problemInstance) {
                case SCHWEFEL:
                    double randomX1Schwefel = (Math.random() * 1000) - 500;
                    double randomX2Schwefel = (Math.random() * 1000) - 500;
                    individuals.add(new Individual(randomX1Schwefel, randomX2Schwefel, populationSize, i, problemInstance));
                    break;
                case HIMMELBLAU:
                    double randomX1Himmelblau = (Math.random() * 12) - 6;
                    double randomX2Himmelblau = (Math.random() * 12) - 6;
                    individuals.add(new Individual(randomX1Himmelblau, randomX2Himmelblau, populationSize, i, problemInstance));
                    break;
                case H1, SCHAFFER:
                    double randomX1H1 = (Math.random() * 200) - 100;
                    double randomX2H1 = (Math.random() * 200) - 100;
                    individuals.add(new Individual(randomX1H1, randomX2H1, populationSize, i, problemInstance));
                    break;
            }
        }
    }

    public Population(int populationSize, Main.ProblemInstance problemInstance, Individual highPotentialIndividual) {
        individuals = new ArrayList<>();
        individuals.add(new Individual(highPotentialIndividual.getX1(), highPotentialIndividual.getX2(), populationSize, 0, problemInstance));

        for (int i = 1; i < populationSize; i++) {
            switch (problemInstance) {
                case SCHWEFEL:
                    double x1Schwefel = (Math.random() * 1000) - 500;
                    double x2Schwefel = (Math.random() * 1000) - 500;
                    individuals.add(new Individual(x1Schwefel, x2Schwefel, populationSize, i, problemInstance));
                    break;
                case HIMMELBLAU:
                    double x1Himmelblau = (Math.random() * 12) - 6;
                    double x2Himmelblau = (Math.random() * 12) - 6;
                    individuals.add(new Individual(x1Himmelblau, x2Himmelblau, populationSize, i, problemInstance));
                    break;
                case H1, SCHAFFER:
                    double x1H1 = (Math.random() * 200) - 100;
                    double x2H1 = (Math.random() * 200) - 100;
                    individuals.add(new Individual(x1H1, x2H1, populationSize, i, problemInstance));
                    break;
            }
        }
    }

    public Population(Population other, Main.ProblemInstance problemInstance) {  //Kopierkonstruktor
        this.individuals = new ArrayList<>();
        for (Individual ind : other.getIndividuals()) {
            this.individuals.add(new Individual(ind.getX1(), ind.getX2(), ind.getAncestry().length(), problemInstance));
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < individuals.size(); i++) {
            Individual individual = individuals.get(i);
            sb.append("Individual ").append(i + 1).append(": ")
                    .append("[x1=").append(individual.getX1())
                    .append(", x2=").append(individual.getX2())
                    .append(", targetValue=").append(individual.getTargetValue())
                    .append("]\n");
        }
        return sb.toString();
    }
}