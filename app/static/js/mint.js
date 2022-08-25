window.addEventListener('load', async () => {
    App.web3 = new Web3(window.ethereum);
    accounts = await ethereum.request({method: 'eth_requestAccounts'}); // get permission to access accounts
    App.start();
});

async function deleteModel() {
    document.getElementById('mint-message').innerHTML = 'Deleting your AI model...';
    document.getElementById('mintBTN').style.display = 'none';
    document.getElementById('mint-progress').style.display = 'initial';

    var model = document.getElementById('modelName').value;
    $.post("/delete_item", {
        name: model
    }, function (err, req, resp) {
        document.getElementById('mint-progress').style.display = 'none';
        document.getElementById('mintBTN').style.display = 'intial';
        window.location.href = "/"
    });
};