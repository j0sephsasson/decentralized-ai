var user = document.getElementById('user');
window.addEventListener('load', function () {
    user.innerHTML = "{{ c_user }}"
});

window.addEventListener('load', async () => {
    App.web3 = new Web3(window.ethereum);
    accounts = await ethereum.request({
        method: 'eth_requestAccounts'
    }); // get permission to access accounts
    App.start();
});

async function useModel() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('utilize-progress').style.display = 'initial';

    var ins = document.getElementById('upload').files;
    var file = ins[0];

    var fd = new FormData();
    fd.append('upload', file);
    fd.append('name', document.getElementById('T-M').value);
    fd.append('row', document.getElementById('row-number').value);

    $.ajax({
        type: "POST",
        url: '/use_model',
        data: fd,
        processData: false,
        contentType: false,
    }).done(function (err, req, resp) {
        if (resp.responseJSON.resp === 'It appears your model was trained on different data.') {
            document.getElementById('utilize-progress').style.display = 'none';
            document.getElementById('results').style.display = 'initial';
            document.getElementById('showResults').innerHTML = resp.responseJSON.resp;
            document.getElementById('show-error').innerHTML = 'Please ensure the data has the same columns as the training data.';
            document.getElementById('show-error').style.display = 'initial';

        } else if (resp.responseJSON.resp === 'No model found with that name.') {
            document.getElementById('utilize-progress').style.display = 'none';
            document.getElementById('results').style.display = 'initial';
            document.getElementById('showResults').innerHTML = resp.responseJSON.resp;
        } else {
            document.getElementById('utilize-progress').style.display = 'none';
            document.getElementById('show-error').style.display = 'none';
            document.getElementById('results').style.display = 'initial';
            document.getElementById('showResults').innerHTML = 'AI Results: ' + resp.responseJSON.resp;
        };

    });
};

function clearInputs() {
    document.getElementById('upload').value = null;
    document.getElementById('upload-label').style.backgroundColor = 'black';
    document.getElementById('T-M').value = null;
    document.getElementById('row-number').value = null;
    document.getElementById('showResults').innerHTML = '';
    document.getElementById('results').style.display = 'none';
};