/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */
package jcompute.opencl.jocl;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.opencl.ClKernel;
import jcompute.opencl.ClMem;
import jcompute.opencl.ClProgram;

public final class ClKernelJocl extends ClKernel {

    @Getter @Accessors(fluent = true) private final cl_kernel id;

    ClKernelJocl(final cl_kernel id, final ClProgram program, final String name) {
        super(program, name);
        this.id = id;
    }

    @Override
    protected int releaseKernel() {
        return CL.clReleaseKernel(id());
    }

    @Override
    public ClKernel setArg(final int argIndex, final ClMem memObj) {
        return setArg(this, argIndex, Sizeof.cl_mem, Pointer.to(((ClMemJocl)memObj).id()));
    }

    @Override
    public ClKernel setArg(final int argIndex, final byte value) {
        return setArg(this, argIndex, 1, Pointer.to(new byte[]{ value }));
    }

    @Override
    public ClKernel setArg(final int argIndex, final short value) {
        return setArg(this, argIndex, Sizeof.cl_short, Pointer.to(new short[]{ value }));
    }

    @Override
    public ClKernel setArg(final int argIndex, final int value) {
        return setArg(this, argIndex, Sizeof.cl_int, Pointer.to(new int[]{ value }));
    }

    @Override
    public ClKernel setArg(final int argIndex, final long value) {
        return setArg(this, argIndex, Sizeof.cl_long, Pointer.to(new long[]{ value }));
    }

    @Override
    public ClKernel setArg(final int argIndex, final float value) {
        return setArg(this, argIndex, Sizeof.cl_float, Pointer.to(new float[]{ value }));
    }

    @Override
    public ClKernel setArg(final int argIndex, final double value) {
        return setArg(this, argIndex, Sizeof.cl_double, Pointer.to(new double[]{ value }));
    }

    /* Set OpenCL kernel argument */
    private ClKernel setArg(final ClKernelJocl kernel, final int argIndex, final long sizeOf, final Pointer pointer) {
        int ret = CL.clSetKernelArg(kernel.id(), argIndex, sizeOf, pointer);
        _Util.assertSuccess(ret, ()->
            String.format("failed to set kernel argument for kernel %s", kernel.name()));
        return kernel;
    }



}
