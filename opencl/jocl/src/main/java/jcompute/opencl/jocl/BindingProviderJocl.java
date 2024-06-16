package jcompute.opencl.jocl;

import jcompute.opencl.ClBinding;
import jcompute.opencl.spi.OpenCLBindingProvider;

public class BindingProviderJocl implements OpenCLBindingProvider {

    @Override
    public ClBinding getBinding() {
        return () -> ClPlatformJocl.listPlatforms();
    }
}
