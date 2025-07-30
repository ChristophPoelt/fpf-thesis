import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import lombok.Getter;

@Getter
public class RunResult {
    private final List<IndividualWithFPF> individualsWithFPF;
    private final int totalGenerations;
    private final double finalBestFitness;
    private final List<Individual> bestIndividualsPerGeneration;

    @JsonCreator
    public RunResult(
            @JsonProperty("individualsWithFPF") List<IndividualWithFPF> individualsWithFPF,
            @JsonProperty("totalGenerations") int totalGenerations,
            @JsonProperty("finalBestFitness") double finalBestFitness,
            @JsonProperty("bestIndividualsPerGeneration") List<Individual> bestIndividualsPerGeneration) {
        this.individualsWithFPF = individualsWithFPF;
        this.totalGenerations = totalGenerations;
        this.finalBestFitness = finalBestFitness;
        this.bestIndividualsPerGeneration = bestIndividualsPerGeneration;
    }
}