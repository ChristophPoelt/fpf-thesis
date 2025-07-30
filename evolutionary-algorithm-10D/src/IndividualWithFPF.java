import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;

@Getter
public class IndividualWithFPF {
    private final double[] genome;
    private final double fpfValue;

    public IndividualWithFPF(Individual individual, double fpfValue) {
        this.genome = individual.getGenome();
        this.fpfValue = fpfValue;
    }

    @JsonProperty("genome")
    public double[] getGenome() {
        return genome;
    }

    @JsonProperty("fpfValue")
    public double getFpfValue() {
        return fpfValue;
    }
}
