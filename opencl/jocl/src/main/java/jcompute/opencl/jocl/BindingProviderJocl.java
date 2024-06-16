package jcompute.opencl.jocl;

import jcompute.opencl.ClBinding;
import jcompute.opencl.spi.OpenCLBindingProvider;

public final class BindingProviderJocl implements OpenCLBindingProvider {

    @Override
    public ClBinding getBinding() {
        return _Jocl.getBinding();
    }
}
